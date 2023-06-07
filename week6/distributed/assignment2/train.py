import os
import torch
from tqdm import tqdm


from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer

from contextlib import nullcontext

from lora_model import LoraModelForCasualLM
from utils.common import download_from_driver
from prepare_data import create_datasets
from torch.distributed import  destroy_process_group


import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

class Trainer:
    def __init__( self,
            model, 
            tokenizer, 
            gpu_id: int, 
            is_ddp_training: bool = True, 
            output_dir: str = 'checkpoints/',  
            num_epochs: int = 10, 
            max_length: int = 128, 
            batch_size: int = 8,
            mixed_precision_dtype =  None,
            gradient_accumulation_steps: int = 16):
        """
        Initialize the Trainer class.

        Args:
            model: Pretrained model object.
            tokenizer: Tokenizer object for text processing.
            num_epochs: Number of training epochs.
            max_length: Maximum sequence length.
            batch_size: Training batch size.
            gpu_id: GPU ID for training.
        """
        

        self.num_epochs = num_epochs
        self.max_length = max_length
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.is_ddp_training = is_ddp_training
        self.gpu_id = gpu_id
        self.model = model.to(f"cuda:{self.gpu_id}")  
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.mixed_precision_dtype = mixed_precision_dtype
        self.ctx  = None
        self.gradscaler = None
        
        # set mixed precision context
        self.set_mixed_precision_context(mixed_precision_dtype)
        
        
    def set_mixed_precision_context(self, mixed_precision_dtype):
        if mixed_precision_dtype is None:
            self.ctx = nullcontext()
        else:
            self.ctx = torch.cuda.amp.autocast(enabled=True, dtype=mixed_precision_dtype)
            if mixed_precision_dtype == torch.float16:
                self.gradscaler = torch.cuda.amp.GradScaler()
            

    def _set_ddp_training(self):
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.gpu_id], output_device=self.gpu_id
        )

        
    def _run_batch(self, batch):
        with self.ctx:
            outputs = self.model(**batch)
            loss = outputs.loss / self.gradient_accumulation_steps
        loss_val = loss.item()

        if self.mixed_precision_dtype == torch.float16:
            self.gradscaler.scale(loss).backward()
        else:
            loss.backward()

        return loss_val

    def _run_epoch(self, train_dataloader, epoch):
        epoch_loss = 0
        self.model.train()

        if _is_master_process():
            train_progress_bar = tqdm(
                train_dataloader, desc=f"Epoch {epoch + 1} [Training]", position=0, leave=False
            )
        else:
            train_progress_bar = train_dataloader

        steps = 0
        self.optimizer.zero_grad()
        for step, batch in enumerate(train_progress_bar):
            steps += 1
            batch = {key: value.to(self.gpu_id) for key, value in batch.items()}
            loss = self._run_batch(batch)
            epoch_loss += loss

            if steps % self.gradient_accumulation_steps == 0:
                if self.mixed_precision_dtype == torch.float16:
                    self.gradscaler.step(self.optimizer)
                    self.gradscaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

        epoch_loss /= len(train_dataloader) / self.gradient_accumulation_steps
        return epoch_loss
    
    def _save_checkpoint(self, epoch):
        checkpoint_path_dir = f"{self.output_dir}/epoch_{epoch}_checkpoint"
        
        # check path_dir exited
        if not os.path.exists(checkpoint_path_dir):
            os.makedirs(checkpoint_path_dir)

        # save checkpoints
        if self.is_ddp_training and _is_master_process():
            # save checkpoints to local
            self.model.module.save_pretrained(checkpoint_path_dir)
        
        else:
            self.model.save_pretrained(checkpoint_path_dir)

    def prepare_dataloader(self, train_dataset, eval_dataset):
        if self.is_ddp_training:
            sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            sampler = None

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, pad_to_multiple_of=self.max_length)

        data_trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            return_tensors="pt",
        )

        data_testloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            sampler=torch.utils.data.SequentialSampler(eval_dataset),
            collate_fn=data_collator,
            return_tensors="pt",
        )

        return data_trainloader, data_testloader
    
    def _eval(self, eval_dataloader, epoch: int):
        avg_loss = 0
        model.eval()
        if _is_master_process():
            eval_progress_bar = tqdm(eval_dataloader, desc=f"Epoch {epoch + 1} [Evaluation]", position=0, leave=False)
        else:
            eval_progress_bar = eval_dataloader
        
        for batch in eval_progress_bar:
            with self.ctx:
                with torch.no_grad():
                    outputs = self.model(**batch) 
            avg_loss += outputs.loss.item()
        avg_loss = avg_loss/(len(eval_dataloader))
        return avg_loss
    
    def run(self, data_path: str, size_valid_set: int = 0.25, seed:int=123):
        """
        Run the training process.

        Returns:
            None
        """
        # Prepare dataset
        train_dataset, eval_dataset = create_datasets(
            tokenizer = self.tokenizer,
            max_length = self.max_length,
            data_path = data_path,
            size_valid_set = size_valid_set,
            seed = seed
           )
        
        train_dataloader, eval_dataloader = self.prepare_dataloader(train_dataset, eval_dataset)
        
        if self.is_ddp_training:
            self._set_ddp_training()

        # Setup the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(self.num_epochs):
            
            if self.is_ddp_training:
                train_dataloader.sampler.set_epoch(epoch)
            
            train_loss = self._run_epoch(train_dataloader, epoch)
            
            if _is_master_process():
                eval_loss = self._eval(eval_dataloader = eval_dataloader, epoch = epoch)
                
                print(f"epoch = {epoch} | avg_train_loss = {train_loss} | eval_loss = {eval_loss}")
                self._save_checkpoint(epoch = epoch)


def load_tokenizer_from_pretrained_model(model_path):
    
    config = AutoConfig.from_pretrained(model_path)
    architecture = config.architectures[0]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if _is_master_process():
        print('Completed to load config & tokenizer')

    if "Llama" in architecture:
        if _is_master_process():
            print("Setting EOS, BOS, UNK, and PAD tokens for LLama tokenizer")
        tokenizer.add_special_tokens(
            {
                "eos_token": "</s>",
                "bos_token": "</s>",
                "unk_token": "</s>",
            }
        )
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
    
    return tokenizer

def _is_master_process():
    ddp_rank = int(os.environ['RANK'])
    return ddp_rank == 0

def load_pretrained_model(local_rank, model_path: str = ""):
    # Load a pretrained AutoModelForCausalLM from the 'model_path' in float16 data type.
    # Make sure to set 'device_map' to '{"": torch.device(f"cuda:{local_rank}")}' for DDP training.

    model = AutoModelForCausalLM.from_pretrained(model_path).to(f"cuda:{local_rank}")

    # Create a LoraConfig with the parameters: r=8, lora_alpha=16,
    # lora_dropout=0.05, bias="none", task_type="CAUSAL_LM".
    # We will then use the config to initialize a LoraModelForCasualLM with the loaded model.
    # Then, print the trainable parameters of the model.

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Create LoRA model
    model = LoraModelForCasualLM(model, lora_config)
    if _is_master_process():
        model.print_trainable_parameters()

    return model


if __name__ == "__main__":
    OUTPUT_DIR = "checkpoints/"
    DRIVER_DATA_PATH = 'https://drive.google.com/file/d/1QpgvQi6mFvN5-6ofmJunDbuz34tlLbLL/view?usp=sharing'

    backend = "nccl"
    model_path = 'bigscience/bloom-1b7'
    if os.environ.get("DEBUG"):
        data_path = 'test_data.json'
    else:
        data_path = 'alpaca_data.json'
        download_from_driver(path= DRIVER_DATA_PATH, location_path= data_path)

    size_valid_set = 0.1
    max_length = 512
    num_epochs = 10
    batch_size = 2
    gradient_accumulation_steps = 16

    learning_rate = 3e-4
    lr_scheduler_type = 'cosine'
    num_warmup_steps = 100
    weight_decay = 0.06

    seed = 0
    log_freq = 1
    eval_freq = 150
    
    # TODO: Choose strategy
    # Choose strategy
    distributed_strategy = "no"

    if distributed_strategy == "ddp":
        # Initialize the process group for distributed data parallelism with nccl backend.
        # After that, you should set the 'local_rank' from the environment variable 'LOCAL_RANK'.

        # Initialize the process group
        torch.distributed.init_process_group(backend=backend)
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        os.environ['RANK'] = '0'
        local_rank = 0

    # Prepare model
    model = load_pretrained_model(local_rank, model_path= model_path)
    # Get tokenizer
    tokenizer = load_tokenizer_from_pretrained_model(model_path = model_path)

    # prepare trainer
    trainer = Trainer(
        model = model, 
        num_epochs = num_epochs,
        max_length = max_length,
        batch_size = batch_size,
        gpu_id=local_rank,
        mixed_precision_dtype = None,  #TODO: Set the mixed precision data type, hint use float16
        tokenizer=tokenizer,
        output_dir= OUTPUT_DIR,
        is_ddp_training = True if distributed_strategy == "ddp" else False,
        gradient_accumulation_steps = gradient_accumulation_steps
    )
    
    # set ddp for wraping model
    # execute trainer 
    trainer.run(
        data_path = data_path,
        size_valid_set = size_valid_set,
        seed =seed
    )

    if distributed_strategy  == "ddp":
        destroy_process_group()

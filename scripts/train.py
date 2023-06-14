import yaml
from scripts.yaml_utils import read_one_block_of_yaml_data
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def train(input_params):

  # *******LOAD YAML FILE DICTIONARY AND BREAK IT DOWN TO VARIABLES******
  args_dict = read_one_block_of_yaml_data(input_params) # To match the yaml file
  
  # select what the model is being finetuned on
  train_col = args_dict['train_col'] # selects what we are finetuning the model on

  # Select dataset
  prcnt = args_dict['data_percent'] # percent of the total data used in dataset
  threshold = args_dict['toxicity_threshold'] # toxicity threshold used for classifier in dataset
  context_length = args_dict['context_length']
  tokenizer_load_path = args_dict['tokenizer_load_path']
  seed = args_dict['seed']

  # create the dataset load path from given hyper-params
  override_dataset_load_path = args_dict['override_dataset_load_path']
  if override_dataset_load_path == True:
    dataset_load_path = args_dict['custom_dataset_load_path']
  else:
    dataset_load_path = "data/data_p"+str(prcnt)+"_sd"+str(seed)+"_thr"+str(threshold)+'_endoftext_cl'+str(context_length)
  print(f'dataset loaded: {dataset_load_path}')

  # given params the model name is picked from dictionary or use the override model name
  override_model_name = args_dict['override_model_name']
  if override_model_name == True:
    model_save_path = 'model/' + args_dict['custom_model_name']
  else:
    model_name_dict = {'texts':'none', 'texts_tox':'tox', 'texts_sentiment':'sen', 'texts_all_labels':'toxsen'}
    model_name = model_name_dict[train_col]
    model_save_path = 'model/model_'+model_name+'_p'+str(prcnt)+'_sd'+str(seed)+'_thr'+str(threshold)+'_endoftext_cl'+str(context_length)
  print(f'Using model save path: {model_save_path}')

  # what model are we finetuning on? This should be 'gpt2'
  model_load_path = args_dict['model_finetuning_from']

  # Training Args
  device = args_dict['device']
  batch_size = args_dict['batch_size']
  epochs_num = args_dict['epochs_num']
  max_number_of_checkpoints = args_dict['max_number_of_checkpoints']
  save_strategy = args_dict['save_strategy']
  save_num_steps = args_dict['save_num_steps']
  logging_steps = args_dict['logging_steps']

  # select if finetuning is continuing from previous checkpoint
  is_from_checkpoint = args_dict['is_from_checkpoint']

  if is_from_checkpoint == True:
    is_from_checkpoint = True
  else:
    is_from_checkpoint = False

  
  #*********LOAD THE TOKENIZER***********
  data_preprocessed = DatasetDict.load_from_disk(dataset_load_path)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)

  # only reading the relevant train_col of our data
  def tokenize(element):
      outputs = tokenizer(
          element[train_col],
          truncation=True,
          max_length=context_length,
          return_overflowing_tokens=False,
          return_length=True,
          padding=True
      )
      return outputs
    
  # tokenise our whole dataset - this does it for train and validation dataset
  tokenized_datasets = data_preprocessed.map(
      tokenize, batched=True, remove_columns=data_preprocessed["train"].column_names
  )
  
  # Load data collector to handle batching
  data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

  #********LOAD THE PRETRAINED MODEL TO FINETUNE ON************
  config = AutoConfig.from_pretrained(
      model_load_path,
      vocab_size=len(tokenizer),
      n_ctx=context_length,
      bos_token_id=tokenizer.bos_token_id,
      eos_token_id=tokenizer.eos_token_id,
  )

  # load a pretrained model instead of a newly initialised model
  model = GPT2LMHeadModel(config).to(device)
  model_size = sum(t.numel() for t in model.parameters())
  print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

  # letting the model know that we have added special tokens
  if model_load_path == 'gpt2':
    model.resize_token_embeddings(len(tokenizer))


  #********LOAD TRAINING ARGS FOR TRAINING**********
  args = TrainingArguments(
      output_dir=model_save_path,
      overwrite_output_dir=False,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      evaluation_strategy="no",
      logging_strategy="steps",
      logging_steps=logging_steps,
      save_strategy=save_strategy,
      save_steps=save_num_steps,
      gradient_accumulation_steps=8,
      num_train_epochs=epochs_num,
      weight_decay=0.1,
      warmup_ratio=0.01,
      lr_scheduler_type="cosine",
      learning_rate=5e-4,
      save_total_limit=max_number_of_checkpoints,
      fp16=True,
      push_to_hub=False,
  )

  trainer = Trainer(
      model=model,
      tokenizer=tokenizer,
      args=args,
      data_collator=data_collator,
      train_dataset=tokenized_datasets["train"],
      eval_dataset=tokenized_datasets["valid"],
  )

  
  #*********START FINE-TUNING**********
  # load from checkpoint if required
  print('\n***********\nTraining Starting\n**************')
  trainer.train(resume_from_checkpoint=is_from_checkpoint)

  #**********SAVE THE FINAL MODEL************
  
  trainer.save_model(model_save_path + '/final_epoch_'+str(epochs_num))
  print('\n***********\nTraining Complete, Model Saved\n**************')




import yaml
import importlib, sys
import numpy as np
import random
from tqdm import tqdm
import random
from scripts.Experiment import Experiment, run_experiment, create_conditions, create_prompts
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from scripts.yaml_utils import read_one_block_of_yaml_data
from tabulate import tabulate


def load_and_run(input_params, random_seed):
  # *******LOAD YAML FILE DICTIONARY AND BREAK IT DOWN TO VARIABLES******

  experiment_dict = read_one_block_of_yaml_data(input_params) # To match the yaml file
  # Set the prompts 
  random.seed(random_seed)

  # Load the validation dataset 
  valid_data_path = experiment_dict['valid_data_path']
  validation_sentences = DatasetDict.load_from_disk(valid_data_path)['valid']
  texts = validation_sentences['texts']

  """ Define stable (not changing in for loops) parameters for experiments """
  # ******* Toxicity threshold *******
  toxicity_threshold = experiment_dict['experiment']['toxicity_threshold'] # This must be alligned to that of the finetuning stage

  # ******* Number of samples ********
  number_of_samples = experiment_dict['experiment']['number_of_samples']

  # ******* Prompting source ****************
  prompt_source = experiment_dict['experiment']['prompt_source'] # 0: Customized prompt / 1: Balanced validation set

  # ******* Set of all models ****************
  model_names=[]
  for i in range(len(experiment_dict['models'])):
    model_names.append(experiment_dict['models'][i]['name'])

  # ******* Set of all cond tokens ****************
  cond_toxic=[]
  cond_sentiment=[]
  for i in range(len(experiment_dict['condition_toxicity'])):
    cond_toxic.append(experiment_dict['condition_toxicity'][i]) # 0 : No condition on toxic / 1: Nontoxic / 2: Toxic
  for i in range(len(experiment_dict['condition_sent'])):
    cond_sentiment.append(experiment_dict['condition_sent'][i])  # 0: No condition on sentiment / 1: Positive / 2: Negative --> SET ZERO FOR NOW

  """ Define all model paths """
  model_paths=[]
  for i in range(len(experiment_dict['models'])):
    model_paths.append(experiment_dict['models'][i]['path'])

  """ Initializing tables and variables """
  # Empty list to store all generated text tables of type DataFrame
  tables_list = []
  # Empty DataFrame to store all generated metrics
  metrics_table = pd.DataFrame(columns=['Model', 'Condition', 'top_p',
                                        'top_k', 'max_new_tokens', 'temperature',
                                        'no_repeat_ngram_size','Total count','Toxic %',
                                        'Nontoxic %', 'Positive %', 'Negative %', 'Toxicity&Sentiment misalignment %'])

  """ Define changing parameters (FOR LOOPS) for experiments """ 

  #Define prompts
  prompts = create_prompts(texts, prompt_source, number_of_samples, prompt='')
  prompts = [string[:-13] for string in prompts]

  #Hyperparams
  top_p=experiment_dict['hyperparameters']['top_p'] # Only chooses words whose probability is >= p % [CAN CHANGE]
  top_k=experiment_dict['hyperparameters']['top_k'] # Chooses next word among the k top prob choices [CAN CHANGE]
  temperature=experiment_dict['hyperparameters']['temperature']
  no_repeat_ngram_size=experiment_dict['hyperparameters']['no_repeat_ngram_size']


  for model_name in model_names: # Iterate through all models
    for model in experiment_dict['models']:
        if model['name'] == model_name:
            path = model['path']
            break
    for cond_tox in cond_toxic: # Iterate through different conditions for toxicity
        for cond_sent in cond_sentiment: # Iterate through different conditions for sentiment
            # Define conditions
            conditions = create_conditions(number_of_samples, cond_tox, cond_sent)
            epoch=1
            # Set the relevant model and tokenizer
            model=path
            tokenizer = path
            print("\n")
            print("Model name: {} | Codition toxicity: {} | Condition sentiment: {} ".format(model_name, cond_tox, cond_sent))
            
            """ Run epxeriment [DO NOT CHANGE anything unless stated] """
            text_table, metrics = run_experiment(model,
                                  model_name,
                                  tokenizer=tokenizer,
                                  number_of_samples=number_of_samples,
                                  toxicity_threshold=toxicity_threshold,
                                  prompts=prompts,
                                  conditions=conditions,
                                  prompt_source=prompt_source,
                                  top_p=top_p, # Only chooses words whose probability is >= p % [CAN CHANGE]
                                  top_k=top_k, # Chooses next word among the k top prob choices [CAN CHANGE]
                                  max_new_tokens=experiment_dict['hyperparameters']['max_length'], # Max number of tokens [CAN CHANGE]
                                  temperature=temperature,
                                  no_repeat_ngram_size=no_repeat_ngram_size,
                                  early_stopping=False,
                                  )
            tables_list.append(text_table)
            metrics_table=pd.concat([metrics_table, metrics], axis=0)

  # Store the metrics_table in .csv file
  model_all_names = '&'.join(model_names) if len(model_names) > 1 else model_names[0]
  condtoxic_all = '&'.join(map(str, cond_toxic)) if len(cond_toxic) > 1 else str(cond_toxic[0])
  condsent_all = '&'.join(map(str, cond_sentiment)) if len(cond_sentiment) > 1 else str(cond_sentiment[0])

  path_output_table = experiment_dict['output_table_path']
  file_name = path_output_table + "/Metrics_table.csv"
  metrics_table.to_csv(file_name, index=False)
  print("Metrics Table saved")

  print("\nPrinting Summary:")

  conditions_to_include = ['','<|nontoxic|>']
  mask = metrics_table['Condition'].isin(conditions_to_include)
  pivot_table_toxicity = metrics_table[mask].pivot_table(
      values='Toxic %', 
      index='Model', 
      columns='Condition'
  )
  conditions_to_include = ['', '<|pos|>']
  mask = metrics_table['Condition'].isin(conditions_to_include)
  pivot_table_sentiment = metrics_table[mask].pivot_table(
      values='Negative %', 
      index='Model', 
      columns='Condition'
  )

  conditions_to_include = ['', '<|nontoxic|><|pos|>']
  mask = metrics_table['Condition'].isin(conditions_to_include)
  pivot_table_both = metrics_table[mask].pivot_table(
      values='Toxicity&Sentiment misalignment %', 
      index='Model', 
      columns='Condition'
  )

  print('\nToxicity misalignment')
  print(tabulate(pivot_table_toxicity, headers='keys', tablefmt='psql'))
  print('\nSentiment misalignment')
  print(tabulate(pivot_table_sentiment, headers='keys', tablefmt='psql')) 
  print('\nToxicity & Sentiment misalignment')
  print(tabulate(pivot_table_both, headers='keys', tablefmt='psql'))
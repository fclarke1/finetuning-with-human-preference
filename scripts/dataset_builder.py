# general
import pandas as pd
import numpy as np
import random
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import sys
from scripts.yaml_utils import read_one_block_of_yaml_data
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from transformers import AutoTokenizer
import os
import sys



def clean_sentence(text):
  # Given a sentence clean it (remove special charecters, etc.)
  clean_sentence = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z ?.'!])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
  return clean_sentence


def remove_sentences_over_length(tokenized_dataset, dataset_name, col_name, df, context_length):
  # find sentences that are within context length
  sentence_lengths = tokenized_dataset[dataset_name]['length']
  sentence_lengths = np.array(sentence_lengths)
  sentence_lengths = sentence_lengths.flatten()

  # get ids of sentences we want to keep
  under_context_length_idxs = (sentence_lengths < context_length)
  print(f'there are {under_context_length_idxs.sum()} sentences under context length')

  # add additional column
  df['in_context_length'] = under_context_length_idxs

  # filter for only True in that new column
  df = df.loc[df['in_context_length'] == True]
  df = df.drop(columns=['in_context_length'])

  return df



def download_and_clean_dataset(yaml_file_name):

  # *******LOAD YAML FILE DICTIONARY AND BREAK IT DOWN TO VARIABLES******
  args_dict = read_one_block_of_yaml_data(yaml_file_name) # To match the yaml file

  # set hyperparameters
  root_path = args_dict['dataset_save_path']  # where the dataset will be saved
  prcnt = args_dict['data_percent']   # percent of total dataset we will use
  seed = args_dict['seed']    # seed used to shuffle the dataset
  threshold = args_dict['toxicity_threshold']   # threshold for toxic classifier

  # file path name
  file_name = str(root_path)+"/data_raw_p"+str(prcnt)+"_sd"+str(seed)+"_thr"+str(threshold)+"_endoftext"

  # *******LOAD DATASET - CAN TAKE 5 MINS******
  dataset = load_dataset(path="tomekkorbak/pile-detoxify")

  # define training data
  training_data = dataset["train"] 

  # describe properties
  print(f"Total data size: {len(training_data)}")
  pprint(training_data.features)

  # seed for reproducibility
  np.random.seed(seed)

  # generate training sample
  indices = np.random.choice(len(training_data), size=int(len(training_data) * prcnt), replace=False)
  training_sample = training_data.select(indices)
  print(f"Training sample size: {len(training_sample)}")

  # remove not needed data
  del training_data
  del dataset

  # make a copy of the original dataset
  data_augmented = copy.deepcopy(training_sample)




  # *******CLEAN SENTENCES******
  texts_col = []

  # iterate over each text and scores in the dataset
  for text in tqdm(training_sample["texts"]):
      text_col = []
      
  # *******CREATE FIRST COLUMN WITH CLEANED SENTENCES (NO CLASSIFIER)******
      for sentence in text:
          cleaned_sentence = clean_sentence(sentence)
          cleaned_sentence += "<|endoftext|>"
          # document
          text_col.append(cleaned_sentence)
      # documents
      texts_col.append(text_col)

  # add texts_tox as a new column to the modified_dataset
  data_augmented = data_augmented.remove_columns("texts")
  data_augmented = data_augmented.add_column("texts", texts_col)



  # *******REMOVE UNWANTED TEXT SOURCES******
  sources_remove = ['Github', 'StackExchange', 'DM Mathematics', 'EuroParl', 'PubMed Central', 'HackerNews', 'FreeLaw', 'ArXiv']
  data_augmented = data_augmented.filter(lambda example: example['meta']['pile_set_name'] not in sources_remove)
  unique_sources = set(example['pile_set_name'] for example in data_augmented['meta'])
  print(unique_sources)

  # remove not needed data
  del texts_col



  # *******ADD COLUMN WITH TOXIC CLASSIFIER AT THE FRONT OF EVERY SENTENCE******
  texts_tox = []

  # iterate over each text and scores in the dataset
  for text, scores in tqdm(zip(data_augmented['texts'], data_augmented["scores"]), total=len(data_augmented)):
      text_tox = []
      
      # iterate over each sentence and score in the text
      for sentence, score in zip(text, scores):
          # prepend token to the sentence based on the score
          if score <= threshold:
              text_tox.append("<|nontoxic|>" + sentence)
          elif score > threshold:
              text_tox.append("<|toxic|>" + sentence)
      
      # append the modified text to texts_tox
      texts_tox.append(text_tox)

  # add texts_tox as a new column to the modified_dataset
  data_augmented = data_augmented.add_column("texts_tox", texts_tox)

  # remove not needed data
  del text_tox

  # don't need this data
  del training_sample



  # *******ADD COLUMN WITH SENTIMENT CLASSIFIER AT THE FRONT OF EVERY SENTENCE******
  
  # initialise classifier
  nltk.download('vader_lexicon')
  analyzer = SentimentIntensityAnalyzer()

  # initialise
  texts_sentiment = []
  texts_sentiment_extreme = []

  # iterate over each document(/text) in the dataset
  for text in tqdm(data_augmented["texts"]):
      text_sentiment = []
      text_sentiment_extreme = []

      # predict sentiment scores
      sentiment_scores = [analyzer.polarity_scores(sentence) for sentence in text]

      # extract sentiment labels
      max_sentiments_extreme = [max(sentiment_score, key=lambda x: sentiment_score[x] if x in ['neg', 'pos', 'neu'] else float('-inf')) for sentiment_score in sentiment_scores]
      max_sentiments = [max(sentiment_score, key=lambda x: sentiment_score[x] if x in ['neg', 'pos'] else float('-inf')) for sentiment_score in sentiment_scores]

      # iterate over each sentence in the document
      for sentence, max_sentiment, max_sentiment_extreme in zip(text, max_sentiments, max_sentiments_extreme):
        # prepend token to the sentence based on the score
        text_sentiment.append("<|"+max_sentiment+"|>" + sentence)
        text_sentiment_extreme.append("<|"+max_sentiment_extreme+"|>")

      # append sentence sentiments to list of documents 
      texts_sentiment.append(text_sentiment)
      texts_sentiment_extreme.append(text_sentiment_extreme)
      
      assert len(text) == len(text_sentiment), \
      "Length of sentiments unequal to document length"

  # add texts_tox as a new column to the modified_dataset
  data_augmented = data_augmented.add_column("texts_sentiment", texts_sentiment)
  data_augmented = data_augmented.add_column("texts_sentiment_extreme", texts_sentiment_extreme)

  # delete this data
  del texts_sentiment
  del texts_sentiment_extreme

  # remove not needed data
  del text_sentiment
  del text_sentiment_extreme



  # *******ADD COLUMN WITH TOXICITY AND SENTIMENT CLASSIFIER AT THE FRONT OF EVERY SENTENCE******
  texts_all_labels = []
  texts_all_labels_extreme = []

  # iterate over documents
  for text, texts_tox, texts_sentiment, texts_sentiment_extreme in tqdm(zip(data_augmented["texts"], data_augmented["texts_tox"], data_augmented["texts_sentiment"], data_augmented["texts_sentiment_extreme"]), total=len(data_augmented)):
    text_all_labels = []
    text_all_labels_extreme = []
    
    # iterate over sentences
    for sentence, text_tox, text_sentiment, text_sentiment_extreme in zip(text, texts_tox, texts_sentiment, texts_sentiment_extreme):
      # identify labels
      toxicity_substr = text_tox.split("<|")[1].split("|>")[0]
      sentiment_substr = text_sentiment.split("<|")[1].split("|>")[0]
      sentiment_substr_extreme = text_sentiment_extreme.split("<|")[1].split("|>")[0]
      # append sentence labels
      text_all_labels.append("<|"+toxicity_substr+"|><|"+sentiment_substr+"|>"+sentence)
      text_all_labels_extreme.append("<|"+toxicity_substr+"|><|"+sentiment_substr_extreme+"|>")
    # append document labels
    texts_all_labels.append(text_all_labels)
    texts_all_labels_extreme.append(text_all_labels_extreme)

  # add texts_all_labels as a new column to the modified_dataset
  data_augmented = data_augmented.add_column("texts_all_labels", texts_all_labels)
  data_augmented = data_augmented.add_column("texts_all_labels_extreme", texts_all_labels_extreme)

  # remove not needed data
  del texts_all_labels
  del texts_all_labels_extreme

  # remove not needed data
  del text_all_labels
  del text_all_labels_extreme



  # *******ADD COLUMN OF CLASS CATEGORY (used later in another script)******
  labels_int_col = []

  # iterate over documents
  for text in tqdm(data_augmented["texts_all_labels_extreme"]):
    labels_int = []
    
    # iterate over sentences
    for sentence in text:
      # identify classified label(s)
      end_index = sentence.find('>', sentence.find('>') + 1)
      label = sentence[0:(end_index+1)]
      # check each label
      if label == "<|nontoxic|><|pos|>": 
        labels_int.append(1)
      elif label == "<|nontoxic|><|neu|>": 
        labels_int.append(2)
      elif label == "<|nontoxic|><|neg|>": 
        labels_int.append(3)
      elif label == "<|toxic|><|pos|>": 
        labels_int.append(4)
      elif label == "<|toxic|><|neu|>": 
        labels_int.append(5)
      elif label == "<|toxic|><|neg|>": 
        labels_int.append(6)    
      else: 
        raise Exception("Label not identified correctly.")

    # append document labels
    labels_int_col.append(labels_int)

  # add texts_all_labels as a new column to the modified_dataset
  data_augmented = data_augmented.add_column("labels_int", labels_int_col)

  # remove not needed data
  del labels_int_col



  # *******CONVERT DATAFRAME INTO A DATASET DICTIONARY******
  data_augm_expand_col = []

  # iterate over each document
  for i, (text, score, texts_tox, texts_sentiment, texts_all_labels, labels_int, source ) in tqdm(enumerate(zip(
      data_augmented['texts'],
      data_augmented['scores'],
      data_augmented['texts_tox'],
      data_augmented['texts_sentiment'],
      data_augmented['texts_all_labels'],
      data_augmented['labels_int'],
      data_augmented['meta']
  ))):
      # iterate over each sentence in the document and append to the expanded data list
      for sentence, score, tox, sentiment, all_labels, label_int in zip(
          text,
          score,
          texts_tox,
          texts_sentiment,
          texts_all_labels,
          labels_int
      ):
          data_augm_expand_col.append({
              'texts': sentence,
              'scores': score,
              'texts_tox': tox,
              'texts_sentiment': sentiment,
              'texts_all_labels': all_labels,
              'labels_int': label_int,
              'source': source['pile_set_name']
          })

  # create the new Dataset object with the expanded data
  data_augm_expand = Dataset.from_dict({
      key: [example[key] for example in data_augm_expand_col]
      for key in data_augm_expand_col[0].keys()
  })

  # remove this data
  del data_augm_expand_col




  # *******REMOVE CLASSIFIER TOKENS FROM 1% OF SENTENCES TO IMPROVE TRAINING******
  # remove data we don't need before making another df
  del data_augmented

  # convert the dataset object to a pandas dataframe
  df_data_augm_expand = data_augm_expand.to_pandas()

  # remove this data
  del data_augm_expand
  
  subset = df_data_augm_expand.sample(frac=0.01, random_state=seed-1)

  # remove all control tokens from all columns
  for col in ['texts_tox',  'texts_sentiment', 'texts_all_labels']:
      subset[col] = subset["texts"]
  subset['labels_int'] = 0

  # update the original DataFrame with the updated subset
  df_data_augm_expand.update(subset)

  # remove not needed data
  del subset



  # *******REMOVE VERY SHORT SENTENCES (LESS THAN 10 WORDS)******
  word_count = df_data_augm_expand['texts'].str.split().str.len()
  word_count[:5]

  # drop anything less than 10 words
  df_data_augm_expand = df_data_augm_expand.drop(df_data_augm_expand[word_count < 10].index)
  df_data_augm_expand.head(10)


  # *******SPLIT INTO TRAIN AND VALIDATION DATASET******
  # randomly shuffle entire dataframe
  df_data_augm_expand_shuffled = df_data_augm_expand.sample(frac=1, random_state=seed+1).reset_index(drop=True)

  # split the indices into training and testing sets
  train_indices, val_indices = train_test_split(df_data_augm_expand.index, test_size=0.01, random_state=seed+2)

  # split the DataFrame based on the indices
  df_train = df_data_augm_expand.loc[train_indices].reset_index(drop=True)
  df_val = df_data_augm_expand.loc[val_indices].reset_index(drop=True)

  del df_data_augm_expand, train_indices, val_indices



  # *******SAVE DATASET******
  # convert your dataframes to Arrow tables
  train_table = Dataset.from_pandas(df_train)
  val_table = Dataset.from_pandas(df_val)

  # del df_train
  # del df_val

  # create a DatasetDict object containing your preprocessed data
  data_preprocessed = datasets.DatasetDict({"train": train_table, "validation": val_table})

  # save to disk
  data_preprocessed.save_to_disk(file_name)





def remove_over_context_length_and_create_valid(yaml_file_name):

  # *******LOAD YAML FILE DICTIONARY AND BREAK IT DOWN TO VARIABLES******
  args_dict = read_one_block_of_yaml_data(yaml_file_name) # To match the yaml file

  # set hyperparameters
  root_path = args_dict['dataset_save_path']  # where the dataset will be saved
  prcnt = args_dict['data_percent']   # percent of total dataset we will use
  seed = args_dict['seed']    # seed used to shuffle the dataset
  threshold = args_dict['toxicity_threshold']   # threshold for toxic classifier
  context_length = args_dict['context_length']    # max length of token sentence
  valid_size = args_dict['validation_size']    # number of sentences in validation set
  valid_max_sentence_length = 17

  dataset_load_path = str(root_path)+"/data_raw_p"+str(prcnt)+"_sd"+str(seed)+"_thr"+str(threshold)+"_endoftext"

  # save file_path
  dataset_save_path = str(root_path)+"/data_p"+str(prcnt)+"_sd"+str(seed)+"_thr"+str(threshold)+"_endoftext_cl"+str(context_length)


  # *******LOAD TOKENIZER******

  # toeknizer uses 'texts_all_labels' because this is the longest text column
  col_name = 'texts_all_labels'

  tokenizer = AutoTokenizer.from_pretrained("tokenizer")

  def tokenize(element):
      outputs = tokenizer(
          element[col_name],
          truncation=True,
          max_length=context_length,
          return_overflowing_tokens=False,
          return_length=True,
          padding=True
      )
      return outputs



  # *******LOAD PREVIOUSLY MADE DATASET AND CONVERT TO DATAFRAME******
  # previously made dataset was full length sentences, we need to remove anyover 64 tokens

  dataset_raw = DatasetDict.load_from_disk(dataset_load_path)
  # convert dataset into df
  df_train = pd.DataFrame(dataset_raw['train'])
  df_valid = pd.DataFrame(dataset_raw['validation'])


  # *******TOKENISE DATASET******
  tokenized_datasets = dataset_raw.map(
      tokenize, batched=False, remove_columns=dataset_raw["train"].column_names
  )

  # remove non needed data
  del dataset_raw


  # *******REMOVE ANY SENTENCES OVER CONTEXT_LENGTH TOKENS******
  df_train = remove_sentences_over_length(tokenized_datasets, 'train', col_name, df_train, context_length)
  df_valid = remove_sentences_over_length(tokenized_datasets, 'validation', col_name, df_valid, context_length)



  # *******MODIFY VALID DATASET******
  # Keep 1000 sentences that are evenly distributed across all classifier types
  # Only keep relatively short sentences (10-17 words long)
  df_valid['word_count'] = df_valid['texts'].str.split().str.len()

  # add label classifiers
  labels_current = []
  for sentence in df_valid["texts_all_labels"]:
    # identify classified label(s)
    end_index = sentence.find('>', sentence.find('>') + 1)
    label = sentence[0:(end_index+1)]
    # check each label
    if label == "<|nontoxic|><|pos|>": 
      labels_current.append(1)
    elif label == "<|nontoxic|><|neu|>": 
      labels_current.append(2)
    elif label == "<|nontoxic|><|neg|>": 
      labels_current.append(3)
    elif label == "<|toxic|><|pos|>": 
      labels_current.append(4)
    elif label == "<|toxic|><|neu|>": 
      labels_current.append(5)
    elif label == "<|toxic|><|neg|>": 
      labels_current.append(6)    
    else: 
      labels_current.append(0)
  df_valid['labels_current'] = labels_current
  
  # Select all sentences in each classifier group and under max_sentence_length
  valid_nontoxic_pos_ids = (df_valid.labels_current == 1) & (df_valid.word_count < valid_max_sentence_length)
  valid_nontoxic_neg_ids = (df_valid.labels_current == 3) & (df_valid.word_count < valid_max_sentence_length)
  valid_toxic_pos_ids = (df_valid.labels_current == 4) & (df_valid.word_count < valid_max_sentence_length)
  valid_toxic_neg_ids = (df_valid.labels_current == 6) & (df_valid.word_count < valid_max_sentence_length)

  # select 250 sentences from each group
  valid_n_split = int(valid_size / 4)
  df_valid_nontoxic_pos = df_valid[valid_nontoxic_pos_ids].sample(n=valid_n_split, random_state=seed)
  df_valid_nontoxic_neg = df_valid[valid_nontoxic_neg_ids].sample(n=valid_n_split, random_state=seed)
  df_valid_toxic_pos = df_valid[valid_toxic_pos_ids].sample(n=valid_n_split, random_state=seed)
  df_valid_toxic_neg = df_valid[valid_toxic_neg_ids].sample(n=valid_n_split, random_state=seed)

  # concatenate all selected sentences together and shuffle
  df_valid_sampled = pd.concat([df_valid_nontoxic_pos, df_valid_nontoxic_neg, df_valid_toxic_pos, df_valid_toxic_neg], ignore_index=True)
  df_valid_sampled = df_valid_sampled.sample(frac=1)

  # check distribution is even
  print('distribution between different labels for valid set:')
  print(df_valid_sampled.labels_current.value_counts())

  df_valid = df_valid_sampled


  # *******CREATE DATASET WITH TRAIN AND VALID******
  under_context_dataset_train = Dataset.from_pandas(df_train)
  under_context_dataset_valid = Dataset.from_pandas(df_valid)

  datasets_under_context = DatasetDict(
      { 
          "train": under_context_dataset_train,
          "valid": under_context_dataset_valid,
      }
  )

  # save dataset
  datasets_under_context.save_to_disk(dataset_save_path)








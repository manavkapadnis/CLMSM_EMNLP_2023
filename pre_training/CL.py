import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import random as rn
# import seaborn as sns
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, precision_recall_curve
import pickle
# import nltk
import math
import os
import sys
import json
import random
import re
# from pandarallel import pandarallel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import (DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, TensorDataset)
import warnings
# import sys
import time
# warnings.filterwarnings('ignore')
# pandarallel.initialize(progress_bar = True)
from tqdm import tqdm
from models_new import TokenTriplet

import wandb
# wandb.login()

os.environ['WANDB_MODE'] = 'dryrun'

start_time = time.time()

SEED = 0
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda'#'cpu'

triplets_path = <PATH OF THE PICKLE FILE THAT HAS TRIPLETS>
init_model_path = <PATH OF THE MODEL TO INITALIZE THE BIDIRECTIONAL ENCODER>

tokenizer = AutoTokenizer.from_pretrained(init_model_path)
    
with open(triplets_path, 'rb') as fr:
  triplets_list = pickle.load(fr)

with open(os.path.join('corpus.pickle'), 'rb') as fr:
  step_corpus = pickle.load(fr)

corpus = [' '.join(item) for item in step_corpus]

seq_len = 512

save_model_folder = 'recipe_triplet_' + init_model_path.split('/')[-1]
project_name = save_model_folder  

wandb.init(project=project_name) 

if os.path.exists(save_model_folder) == False:
  os.mkdir(save_model_folder)

text = [corpus[item[0]] for item in triplets_list]
text_pos = [corpus[item[1]] for item in triplets_list]
text_neg = [corpus[item[2]] for item in triplets_list]

print(len(text), len(text_pos), len(text_neg), len(triplets_list))

def make_dataset(sents1, sents2, sents3, tokenizer, max_len_input):        

  all_input_ids1 = []
  all_input_ids2 = []
  all_input_ids3 = []
  all_attention_masks1 = []
  all_attention_masks2 = []
  all_attention_masks3 = []
  if 'roberta' not in init_model_path:
    all_tok_type_ids_1 = []
    all_tok_type_ids_2 = []
    all_tok_type_ids_3 = []

  for sent1, sent2, sent3 in zip(sents1, sents2, sents3):

    encoded_input1 = tokenizer(sent1, max_length = max_len_input, padding = 'max_length', truncation = True)
    encoded_input2 = tokenizer(sent2, max_length = max_len_input, padding = 'max_length', truncation = True)
    encoded_input3 = tokenizer(sent3, max_length = max_len_input, padding = 'max_length', truncation = True)

    # print(len(encoded_input1['input_ids']))
    # print(len(encoded_input2['input_ids']))

    all_input_ids1.append(encoded_input1['input_ids'])
    all_input_ids2.append(encoded_input2['input_ids'])
    all_input_ids3.append(encoded_input3['input_ids'])

    all_attention_masks1.append(encoded_input1['attention_mask'])
    all_attention_masks2.append(encoded_input2['attention_mask'])
    all_attention_masks3.append(encoded_input3['attention_mask'])

    if 'roberta' not in init_model_path:
      all_tok_type_ids_1.append(encoded_input1['token_type_ids'])
      all_tok_type_ids_2.append(encoded_input2['token_type_ids'])
      all_tok_type_ids_3.append(encoded_input3['token_type_ids'])

  all_input_ids1 = torch.as_tensor(all_input_ids1)
  all_input_ids2 = torch.as_tensor(all_input_ids2)
  all_input_ids3 = torch.as_tensor(all_input_ids3)  

  print(len(all_input_ids1), len(all_input_ids2), len(all_input_ids3))

  all_attention_masks1 = torch.as_tensor(all_attention_masks1)
  all_attention_masks2 = torch.as_tensor(all_attention_masks2)
  all_attention_masks3 = torch.as_tensor(all_attention_masks3)  
  if 'roberta' not in init_model_path:
    all_tok_type_ids_1 = torch.as_tensor(all_tok_type_ids_1)
    all_tok_type_ids_2 = torch.as_tensor(all_tok_type_ids_2)
    all_tok_type_ids_3 = torch.as_tensor(all_tok_type_ids_3)


  if 'roberta' in init_model_path:
    dataset = TensorDataset(all_input_ids1, all_input_ids2, all_input_ids3, all_attention_masks1, all_attention_masks2, all_attention_masks3)#, timestep_indices1, timestep_indices2, timestep_indices3)
  else:
    dataset = TensorDataset(all_input_ids1, all_input_ids2, all_input_ids3, all_attention_masks1, all_attention_masks2, all_attention_masks3, all_tok_type_ids_1, all_tok_type_ids_2, all_tok_type_ids_3)#, timestep_indices1, timestep_indices2, timestep_indices3)    
  return dataset



train_dataset = make_dataset(text, text_pos, text_neg, tokenizer, seq_len)

print()
print(len(train_dataset))

from transformers import AutoModel
# from hier_utils import custom_loss_triplet

EPOCHS = int(sys.argv[6])

if 'large' in init_model_path.lower():
  BATCH_SIZE = 8

  accum_iter = 4
else:

  BATCH_SIZE = 16

  accum_iter = 2

model = nn.DataParallel(TokenTriplet(model_path=init_model_path))

model.to(device);



train_dataloader = DataLoader(train_dataset,BATCH_SIZE,shuffle=True, num_workers=8)

len_dataloader = len(train_dataloader)

print()
print('Loaded all data')
print()
print()

scores = []
loss_list = []
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = math.ceil(total_steps/accum_iter))

wandb.define_metric("custom_step")
wandb.define_metric("triplet/loss", step_metric='custom_step')
wandb.define_metric("LearningRate", step_metric='custom_step')

for epoch_i in tqdm(range(EPOCHS)):
  total_train_loss = 0
  model.train()
  epoch_iterator = tqdm(train_dataloader, desc="Iteration")

  model.zero_grad()
  
  for step, batch in enumerate(epoch_iterator):
    # model.zero_grad()
    if 'roberta' in init_model_path:
      anchor_out, pos_out, neg_out = model.forward(batch[0].to(device), batch[1].to(device), batch[2].to(device),batch[3].to(device), batch[4].to(device), batch[5].to(device))#, batch[-3].to(device), batch[-2].to(device), batch[-1].to(device))
    else:
      anchor_out, pos_out, neg_out = model.forward(batch[0].to(device), batch[1].to(device), batch[2].to(device),batch[3].to(device), batch[4].to(device), batch[5].to(device),batch[6].to(device), batch[7].to(device), batch[8].to(device))#, batch[-3].to(device), batch[-2].to(device), batch[-1].to(device))            
    
    loss = criterion(anchor_out.to(device), pos_out.to(device), neg_out.to(device))
    loss = loss / accum_iter

    if (epoch_i * len_dataloader+step+1)%50==0:
      wandb_dict={
        'triplet/loss': loss.mean().item(),
        'custom_step': step+1,
        'LearningRate': scheduler.get_lr()[0]
      }
      wandb.log(wandb_dict)

    total_train_loss += loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_dataloader)):
      optimizer.step()
      model.zero_grad()
      scheduler.step()
#     if (epoch_i * len(train_dataloader) + step + 1) % 10000 == 0:
#       torch.save(model.state_dict(), os.path.join(save_model_folder, 'triplet_epochs_{}_{}_{}.pt'.format(epoch_i + 1, step + 1, BATCH_SIZE)))
  avg_train_loss = total_train_loss / len(train_dataloader)
  loss_list.append(avg_train_loss)
  print('Loss after epoch {} = {}'.format(epoch_i + 1, avg_train_loss))

  print()
  print('Epoch {}'.format(epoch_i + 1))
  print('Time taken (in hrs.)')
  print(np.round((time.time() - start_time)/3600, 5))

  torch.save(model.state_dict(), os.path.join(save_model_folder, 'triplet_epochs_{}.pt'.format(epoch_i + 1)))

wandb.finish()
print(loss_list)

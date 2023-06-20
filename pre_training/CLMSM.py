import os
import sys
import pickle
import numpy as np
import random as rn
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForTokenClassification
from torch.utils.data import (DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler, TensorDataset)

from tqdm import tqdm

import wandb
# wandb.login()

os.environ['WANDB_MODE'] = 'dryrun'

SEED = 0
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda'

model_path = sys.argv[1]

if '/scratch' in model_path:
	project_name = 'mask_step_pretraining_plus_contr_{}'.format(model_path.split('/')[-2])
else:
	project_name = 'mask_step_pretraining_plus_contr_{}'.format(model_path.split('/')[-1])

if 'roberta-large' in model_path.lower():
	model_type = 'roberta-base'
elif 'roberta' in model_path.lower():
	model_type = 'roberta-base'
elif 'spanbert' in model_path.lower():
	model_type = 'spanbert-base-cased'
else:
	model_type = 'bert-base-uncased'

with open('masked_step_plus_contr_recipe_pretrain_INP_corpus_{}.pickle'.format(model_type), 'rb') as fr:
	input_set = pickle.load(fr)

with open('masked_step_plus_contr_recipe_pretrain_OUT_corpus_{}.pickle'.format(model_type), 'rb') as fr:
	target_set = pickle.load(fr)

with open(<TRIPLETS_PICKLE_FILE>, 'rb') as fr:
	triplets_list = pickle.load(fr)

if flip_anch_pos == 'flip':
	triplets_list_new = [[item[1], item[0], item[2]] for item in triplets_list]
	triplets_list = triplets_list + triplets_list_new

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path, num_labels = tokenizer.vocab_size)

'''
model_max_len = tokenizer.model_max_length
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
mask_token_id = tokenizer.mask_token_id
'''

pad_token_id = tokenizer.pad_token_id
out_label_to_ignore = -100	

def get_attn_mask(inputs):
	attn_mask = []
	for recipe_token_ids in inputs:
		tmp_attn_mask = []
		for token_id in recipe_token_ids:
			if token_id == pad_token_id:
				tmp_attn_mask.append(0)
			else:
				tmp_attn_mask.append(1)

		attn_mask.append(tmp_attn_mask)
	return attn_mask

inputs_1= []
inputs_2= []
inputs_3= []
attn_mask_1= []
attn_mask_2= []
attn_mask_3= []
targets_1= []
targets_2= []
targets_3= []

for triplet in triplets_list:
	if ((triplet[0] in input_set) and (triplet[1] in input_set) and (triplet[2] in input_set))==False:
		continue
	inputs_1.append(input_set[triplet[0]])
	inputs_2.append(input_set[triplet[1]])
	inputs_3.append(input_set[triplet[2]])

	targets_1.append(target_set[triplet[0]])
	targets_2.append(target_set[triplet[1]])
	targets_3.append(target_set[triplet[2]])

attn_mask_1 = get_attn_mask(inputs_1)
attn_mask_2 = get_attn_mask(inputs_2)
attn_mask_3 = get_attn_mask(inputs_3)

inputs_1 = torch.tensor(np.array(inputs_1))#, dtype=torch.long)
inputs_2 = torch.tensor(np.array(inputs_2))#, dtype=torch.long)
inputs_3 = torch.tensor(np.array(inputs_3))#, dtype=torch.long)

attn_mask_1 = torch.tensor(np.array(attn_mask_1))#, dtype=torch.long)
attn_mask_2 = torch.tensor(np.array(attn_mask_2))#, dtype=torch.long)
attn_mask_3 = torch.tensor(np.array(attn_mask_3))#, dtype=torch.long)

targets_1 = torch.tensor(np.array(targets_1))#, dtype=torch.long)
targets_2 = torch.tensor(np.array(targets_2))#, dtype=torch.long)
targets_3 = torch.tensor(np.array(targets_3))#, dtype=torch.long)

# attn_mask = torch.tensor(np.array(attn_mask))#, dtype=torch.long)
# targets = torch.tensor(np.array(targets))#, dtype=torch.long)

print(inputs_1.shape, targets_1.shape, attn_mask_1.shape)
print(inputs_2.shape, targets_2.shape, attn_mask_2.shape)
print(inputs_3.shape, targets_3.shape, attn_mask_3.shape)

train_dataset = TensorDataset(inputs_1,attn_mask_1,targets_1,inputs_2,attn_mask_2,targets_2,inputs_3,attn_mask_3,targets_3)

print(len(train_dataset))

EPOCHS = 1

if 'large' in model_path.lower():
	BATCH_SIZE = 8 # same as triplet systems

	accum_iter = 4
else:
	BATCH_SIZE = 16 # same as triplet systems

	accum_iter = 2

lambda_ = 0.05

train_dataloader = DataLoader(train_dataset,BATCH_SIZE,shuffle=True, num_workers=8)

len_dataloader = len(train_dataloader)

print(len_dataloader)

wandb.init(project=project_name)

save_model_folder = project_name

if os.path.exists(save_model_folder)==False:
	os.mkdir(save_model_folder)

model = nn.DataParallel(AutoModelForTokenClassification.from_pretrained(model_path, config = config))
model.to(device);

criterion_contr = nn.TripletMarginLoss(margin=1.0, p=2)

optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

total_steps = len_dataloader * EPOCHS

num_lr_decay_steps = total_steps/accum_iter

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = num_lr_decay_steps)

wandb.define_metric("custom_step")
wandb.define_metric("loss", step_metric='custom_step')
wandb.define_metric("triplet/loss", step_metric='custom_step')
wandb.define_metric("maskstep/loss", step_metric='custom_step')
wandb.define_metric("lambda", step_metric='custom_step')                                            
wandb.define_metric("LearningRate", step_metric='custom_step')

def save_parallel_model(model, path):
	for child in model.children():
		child_model = child
	# if is_correct:
	# 	child_model.model_layer.save_pretrained(path)
	# else:
	child_model.save_pretrained(path)

loss_list = []

for epoch_i in tqdm(range(EPOCHS)):
	total_train_loss = 0
	model.train()
	epoch_iterator = tqdm(train_dataloader, desc="Iteration")

	model.zero_grad()

	for step, batch in enumerate(epoch_iterator):
	# for step in tqdm(range(len_dataloader), desc="Iteration"):
		# batch = unpickler.load()
		# if is_correct:
		# 	if only_step_loss:
		# 		out_step=model(batch[0].to(device), batch[1].to(device))
		# 	elif only_token_loss:
		# 		out_token=model(batch[0].to(device), batch[1].to(device))
		# 	else:
		# 		out_step, out_token = model(batch[0].to(device), batch[1].to(device))
		# 	if only_token_loss==False:
		# 		step_loss = criterion(torch.transpose(out_step, -1, -2), batch[2].to(device))
		# 	if only_step_loss==False:
		# 		token_loss = criterion(torch.transpose(out_token, -1, -2), batch[3].to(device))
		# else:
		model_out_1 = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), labels=batch[2].to(device), output_hidden_states = True)#.loss
		model_out_2 = model(input_ids=batch[3].to(device), attention_mask=batch[4].to(device), labels=batch[5].to(device), output_hidden_states = True)#.loss
		model_out_3 = model(input_ids=batch[6].to(device), attention_mask=batch[7].to(device), labels=batch[8].to(device), output_hidden_states = True)#.loss

		mask_step_loss = model_out_1.loss + model_out_2.loss + model_out_3.loss
			# token_loss = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), labels=batch[3].to(device)).loss
		out_1 = model_out_1.hidden_states[-1][:,0,:]
		out_2 = model_out_2.hidden_states[-1][:,0,:]
		out_3 = model_out_3.hidden_states[-1][:,0,:]

		triplet_loss = criterion_contr(out_1.to(device), out_2.to(device), out_3.to(device))

		# if only_token_loss==False:
		# 	step_loss = step_loss/accum_iter
		# if only_step_loss==False:
		# 	token_loss = token_loss/accum_iter

		if mask_step_loss.dim() > 0:
			mask_step_loss = mask_step_loss.mean() # multiple GPUs - loss tensor is the output

		loss = triplet_loss + lambda_ * mask_step_loss # mask step loss range would be 6-7.5, triplet loss does not exceed 0.4

		loss = loss/accum_iter


		# print(loss)

		if (epoch_i * len_dataloader+step+1)%500==0:
			wandb_dict = {
				'triplet/loss': triplet_loss.item(),
				'maskstep/loss': mask_step_loss.item(),
				'loss': loss.item(),
				'lambda': lambda_,
				'custom_step': step+1,
				'LearningRate': scheduler.get_lr()[0]
			}

			wandb.log(wandb_dict)

		# loss = loss / accum_iter

		total_train_loss += loss.item()

		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


		if ((step + 1) % accum_iter == 0) or (step + 1 == len_dataloader):
			optimizer.step()
			model.zero_grad()
			scheduler.step()
			# print()
			# print('Model')
			# print(model)
			# print(model.__dir__())
			# torch.save(model.state_dict(), os.path.join(save_model_folder, 'model_global_step_{}.pt'.format(epoch_i * len_dataloader + step + 1)))
			# save_parallel_model(model, save_model_folder + '/' +  'model_global_step_{}.pt'.format(epoch_i * len_dataloader + step + 1))
			# print(1/0)				
# 		if (epoch_i * len_dataloader + step + 1) % (14000*2) == 0:
			# save_parallel_model(model, os.path.join(save_model_folder, 'model_global_step_{}'.format(epoch_i * len_dataloader + step + 1)))
# 			save_parallel_model(model, os.path.join(save_model_folder, 'model_global_step_{}'.format(epoch_i * len_dataloader + step + 1)))

	avg_train_loss = total_train_loss / len_dataloader
	loss_list.append(avg_train_loss)
	print('Loss after epoch {} = {}'.format(epoch_i + 1, avg_train_loss))

	# print()
	# print('Epoch {}'.format(epoch_i + 1))
	# print('Time taken (in hrs.)')
	# print(np.round((time.time() - start_time)/3600, 5))

	# torch.save(model.state_dict(), os.path.join(save_model_folder, 'model_epochs_{}.pt'.format(epoch_i + 1)))

	save_parallel_model(model, os.path.join(save_model_folder, 'model_epochs_{}'.format(epoch_i + 1)))

wandb.finish()

print(loss_list)

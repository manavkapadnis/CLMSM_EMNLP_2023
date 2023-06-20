import pandas as pd
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
import sys
import os

os.environ['WANDB_MODE'] = 'dryrun'

MODEL_TYPE = sys.argv[1]
MODEL_NAME = sys.argv[2]

import json
with open('train-v2.0.json', 'r') as f:
    train_data = json.load(f)

train_data = [item for topic in train_data['data'] for item in topic['paragraphs'] ]
# print(train_data[0])
# train_data = train_data[:5000]

special_name = MODEL_NAME.replace('/model_epochs_1', '').split('/')[-1]

if 'embert' in MODEL_NAME:
    special_name = 'embert_' + special_name
elif 'emroberta' in MODEL_NAME:
    special_name = 'emroberta_' + special_name

output_parent_dir = 'output_{}'.format(special_name)
cache_parent_dir = 'cache_dir_{}'.format(special_name)
project_name = 'SQuADv2_{}'.format(special_name)

train_args = QuestionAnsweringArgs()

train_args.cache_dir = cache_parent_dir
train_args.output_dir = output_parent_dir
train_args.wandb_project = project_name
train_args.best_model_dir = '{}/best_model'.format(train_args.output_dir)

train_args.max_answer_length = 30    
train_args.learning_rate = 5e-5
train_args.num_train_epochs = 3
train_args.overwrite_output_dir= True
train_args.reprocess_input_data= False
if MODEL_TYPE == 'longformer':
    train_args.train_batch_size= 6
    train_args.gradient_accumulation_steps = 8
else:
    train_args.train_batch_size= 48
if 'large' in MODEL_NAME.lower():
    train_args.n_gpu=4

train_args.fp16= False
#       'wandb_project': "simpletransformers"


model = QuestionAnsweringModel(MODEL_TYPE, MODEL_NAME, args=train_args)
model.train_model(train_data)

os.system('rm -r {}'.format(cache_parent_dir))

for item in os.listdir(output_parent_dir):
    if '-epoch-{}'.format(train_args.num_train_epochs) in item:
        continue
    item_path = os.path.join(output_parent_dir, item)
    os.system('rm -r {}'.format(item_path))


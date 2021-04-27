# imports 
from pegasus_students import create_student_with_configuration, shift_tokens_right
import logging
from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration,PegasusConfig,AutoTokenizer,AutoModelForSeq2SeqLM
import datasets

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F

from typing import Callable, Dict, Iterable, List, Tuple, Union
from transformers import EvalPrediction, PreTrainedTokenizer

from transformers import AdamW
import wandb
import gc
from data_loader import prepare_data
import numpy as np
from utils import calculate_rouge, freeze
from loss_functions import blended_loss


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

wandb.init()

tokenizer = AutoTokenizer.from_pretrained('google/pegasus-gigaword')

teacher = AutoModelForSeq2SeqLM.from_pretrained('/mnt/d/Pegasus_Model/saves/Pegasus_gigaword')
student = create_student_with_configuration(teacher,
                                      e=4,
                                      d=4,
                                      copy_first_teacher_layers = False,
                                      save_path='./student')
wandb.watch(student)
gc.collect()

dataset = datasets.load_from_disk('/mnt/d/Pegasus_Model/saves/gigaword')

#source data
train_texts, train_labels = dataset['train']['document'][:100000], dataset['train']['summary'][:100000]
valid_texts, valid_labels = dataset['validation']['document'][:10000], dataset['validation']['summary'][:10000]
test_texts, test_labels = dataset['test']['document'][:1000], dataset['test']['summary'][:1000]
train_dataset, test_dataset, valid_dataset = prepare_data('google/pegasus-gigaword', train_texts, train_labels,valid_texts, valid_labels,test_texts, test_labels)

del dataset 
gc.collect()

train_dataloader = DataLoader(train_dataset,batch_size=24,num_workers = 2)
test_dataloader = DataLoader(test_dataset,batch_size=24,num_workers = 2)
validation_dataloader = DataLoader(valid_dataset,batch_size=24,num_workers = 2)

optimizer = AdamW(student.parameters(), lr=5e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

pad_token_id = tokenizer.pad_token_id


print(torch.cuda.get_device_name(0))
print('Memory Usage:')
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

student.to('cuda')
teacher.to('cuda')

print(torch.cuda.get_device_name(0))
print('Memory Usage:')
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# ----------------
# TRAINING LOOP
# ----------------
num_epochs = 2
for epoch in range(num_epochs):
  i = 0
  # TRAINING LOOP
  for train_batch in train_dataloader:
    i=i+1
    y = train_batch['labels'].to('cuda')
    x = {
        'input_ids':train_batch['input_ids'].to('cuda'),
         'attention_mask':train_batch['attention_mask'].to('cuda')
    }
    #print(y , x)
    decoder_input_ids = shift_tokens_right(y, pad_token_id)
    
    
    student.train(True)

    logits = student(x['input_ids'],
            attention_mask=x['attention_mask'],
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False)
  
    loss = blended_loss(teacher,student,x,y,pad_token_id)

    print(f'epoch|{epoch} iteration {i} train loss: {loss.item()}')

    loss.backward()

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    wandb.log({"loss": loss})

    gc.collect()
    torch.cuda.empty_cache()


  # VALIDATION LOOP
  with torch.no_grad():
    val_loss = []
    student.eval()
    for val_batch in validation_dataloader:
        y = val_batch['labels'].to('cuda')
        x = {
              'input_ids':val_batch['input_ids'].to('cuda'),
              'attention_mask':val_batch['attention_mask'].to('cuda')
          } 
        decoder_input_ids = shift_tokens_right(y, pad_token_id)
            
        logits = student(x['input_ids'],
              attention_mask=x['attention_mask'],
              decoder_input_ids=decoder_input_ids,
              output_hidden_states=True,
              output_attentions=False,
              use_cache=False)
            
        val_loss.append(blended_loss(teacher,student,x,y,pad_token_id).item())

    val_losses = torch.mean(torch.tensor(val_loss))
    print('val_loss: ', val_losses.item())


  # Evaluation LOOP
  with torch.no_grad():
    student.eval()
    all_labels = []
    all_preds = []
    test_loss=[]
    for test_batch in test_dataloader:
        y = test_batch['labels'].to('cuda')
        x = {
                'input_ids':test_batch['input_ids'].to('cuda'),
                'attention_mask':test_batch['attention_mask'].to('cuda')
            } 
          
        prediction = student.generate(**x)
      
        all_labels.append(y)
        all_preds.append(prediction)
              
        test_loss.append(blended_loss(teacher,student,x,y,pad_token_id).item())

    test_losses = torch.mean(torch.tensor(val_loss))
    print('test_loss: ', test_losses.item())
    preds = [tokenizer.decode(pred[0]) for pred in all_preds]
    lbls = [tokenizer.decode(lbl[0]) for lbl in all_labels]
    rouge_score = calculate_rouge(pred_lns=preds,tgt_lns=lbls)
    wandb.log(rouge_score)
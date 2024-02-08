import pytorch_lightning as pl
import torch
import numpy
import pandas
import seaborn as sns
from transformers import AutoTokenizer


tokens = ['This is a cell', 'Cell is good',
          'good is bad', 'to be or not to be', 'let it be']
print(len(tokens))

tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base')

tokenized = tokenizer(tokens, padding=True, return_tensors='pt').to('cpu')
print(tokenized)

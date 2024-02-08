# %%
import datasets
from LLM4Bio.Geneformer.pretrainer import GeneformerPreCollator
import pickle
from transformers import DataCollator, DataCollatorForLanguageModeling
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
from LLM4Bio.models import TextEncoder


tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/LinkBERT-large')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
model = TextEncoder()
model(inputs).shape

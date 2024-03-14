# %%
import numpy as np
import os
from datetime import date
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import json
import pickle
from LLM4Bio.data import LLM4Bio_data
from LLM4Bio.models import TextGeneContrastive
from LLM4Bio.embed import Embedder
from LLM4Bio.zero_shot_classification import classify, classify_genes
from LLM4Bio.gene_perturb import perturb
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from LLM4Bio.Geneformer.pretrainer import GeneformerPreCollator

save_dir = './saves'
data_dir = './data'
config = {
    'emb_dim': 1024,
    'freeze_text_model': True,
    'freeze_gene_model': True,
    'use_cell_type': False,
    'lr': 1e-3,
    'batch_size': 16,
    'n_top_genes': 50,
    'dino_nlayers': 3,
    'data_dir': data_dir,
    'save_dir': save_dir,
    'text_model': 'biolinkbert',
    'gene_model': 'geneformer',
    'gene_dataset': 'PBMC',
    'gene_summary': 'NCBI',
    'cell_ontology': 'mixed',
}

dataset = LLM4Bio_data(config)

dataset.prepare_data()
dataset.setup('')

# [    0  4997  1720  5873 12698 16906 14127 17585   396 19629]

dl = dataset.test_dataloader()
model = TextGeneContrastive(config)
model.build_summary_table(
    dataset.get_summaries(mode='gene' if not config['use_cell_type'] else 'gene_cell', tokenized=True))

emb, ct, gene = perturb(model, dl)

pred_g, predct = classify_genes(emb, model.encode_summaries(
    dataset.get_summaries('gene_cell', True, True)), 'gene_cell')

print(pred_g)

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
from LLM4Bio.zero_shot_classification import classify

save_dir = './saves'
data_dir = './data'
config = {
    'emb_dim': 1024,
    'freeze_text_model': True,
    'freeze_gene_model': True,
    'use_cell_type': False,
    'lr': 1e-3,
    'batch_size': 16,
    'n_top_genes': 7,
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

model = TextGeneContrastive(config)
model.build_summary_table(
    dataset.get_summaries(mode='gene' if not config['use_cell_type'] else 'gene_cell', tokenized=True))
embedder = Embedder(dataset.token_dictionary,
                    dataset.cell2index,
                    dataset.gene2ensembl)
loader = dataset.test_dataloader()

embedded = embedder.get_embed(model, loader, 10, include=[
                              'gene_emb', 'text_emb', 'cell_emb_gene', 'cell_emb_text'])

encoded_summaries = model.encode_summaries(
    dataset.get_summaries('gene', tokenized=True, use_names=True))
truect, predct = classify(
    embedded, encoded_summaries, mode='gene', max_genes_per_cell=2)

print(truect.shape, predct.shape)

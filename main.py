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
from LLM4Bio.perturb import perturb
from LLM4Bio.zero_shot_classification import classify
from sklearn import svm
save_dir = './saves'
data_dir = './data'
config = {
    'emb_dim': 1024,
    'freeze_text_model': True,
    'loss_type': 'gene_celltype',
    'freeze_gene_model': True,
    'use_cell_type': True,
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
    dataset.get_summaries(mode=config['loss_type']))


embs, ct, genes = perturb(model, dataset.test_dataloader(), n_cells=100, keys=[
                          'gene_enc', 'geneformer_encoded'])


print(embs['gene_enc'].shape, ct.shape,
      genes.shape, embs['geneformer_encoded'].shape)

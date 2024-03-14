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


selected_genes = ['ENSG00000107317', 'ENSG00000101162', 'ENSG00000197561']
selected_genes = [dataset.token_dictionary[g] for g in selected_genes]


model = TextGeneContrastive(config)
model.build_summary_table(
    dataset.get_summaries(mode=config['loss_type']))

emb, ct, g = perturb(model, dataset.val_dataloader())
print(emb.shape, ct.shape, g.shape)
# embedder = Embedder(dataset.token_dictionary, dataset.cell2index,
#                     dataset.gene2ensembl)
# embedded_data = embedder.get_embed(
#     model, dataset.val_dataloader(), n_cells=100)


# embs, ct, gene = embedded_data.get_all_gene_embedding()
# print(gene[:10])
# print(ct[:10])
# encoded_summaries = model.encode_summaries(
#     dataset.get_summaries('gene', use_names=True))
# gene_pred = classify(embs, encoded_summaries, mode='gene')
# print(gene_pred[:10])

# encoded_summaries = model.encode_summaries(
#     dataset.get_summaries('concat_celltype', use_names=True))
# cell_pred, gene_pred = classify(embs, encoded_summaries, mode='concat')
# print(gene_pred.shape, cell_pred.shape)
# print(cell_pred[:10], gene_pred[:10])
# encoded_summaries = model.encode_summaries(
#     dataset.get_summaries('concat_celltype', use_names=True), dict_key='cell')
# gene_pred = classify(embs, encoded_summaries, mode='cell')
# print(gene_pred.shape)
# print(gene_pred[:10])
# # # %%
# logger = TensorBoardLogger("temp", name="my_model")
# trainer = Trainer(max_epochs=100, logger=logger)
# trainer.fit(model, train_dataloaders=dataset.train_dataloader(),
#             val_dataloaders=dataset.val_dataloader())

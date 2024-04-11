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
hold_out_genes = ['GP6', 'PASK', 'PTGS1', 'TXK', 'DUSP2', 'COL19A1', 'SERPING1', 'DAB2', 'IFNG',
                  'UGCG', 'BEND2', 'CD83', 'KIR3DL2', 'CDK12', 'MNDA', 'CEP78', 'CCDC50', 'PRDM1',
                  'GAS6', 'CD8B'] + ['S100A8', 'GNLY', 'NKG7', 'MS4A1', 'CD8A']
hold_out_celltypes = ['Erythrocytes',
                      'Plasmacytoid dendritic cells', 'CD10+ B cells']

hold_out_genes = []
hold_out_celltypes = []
config = {
    'emb_dim': 256,
    'freeze_text_model': True,
    'freeze_gene_model': True,
    'use_cell_type': False,
    # 'concat_celltype', 'gene_celltype', 'concat', 'gene'
    'loss_type': 'concat',
    'flatten': True,  # flattens cells and then compute clip
    'lr': 1e-2,
    'lr_schedule': True,
    'concat_option': 2,
    'batch_size': 16,
    'n_top_genes': 7,
    'use_bn': False,
    'use_dr': True,
    'dr_rate': 0.2,
    'leave_out_celltypes': hold_out_celltypes,
    'leave_out_genes': hold_out_genes,
    'text_agumentations': [],
    'temperature': 0.01,
    'dino_nlayers': 3,
    'data_dir': data_dir,
    'save_dir': save_dir,
    'text_model': 'BioLinkBERT-base',
    'gene_model': 'geneformer',
    'gene_dataset': 'PBMC',
    'gene_summary': 'NCBI',
    'cell_ontology': 'mixed',
}
config['use_bert_encoded'] = True if len(
    config['text_agumentations']) == 0 and config['freeze_text_model'] else False
# config['use_bert_encoded'] = False
# config['use_bert_encoded'] = True

dataset = LLM4Bio_data(config)

dataset.prepare_data()
dataset.setup('')


model = TextGeneContrastive(config)


trainer = Trainer(max_epochs=50)
trainer.fit(model, train_dataloaders=dataset.train_dataloader(),
            val_dataloaders=dataset.val_dataloader())

# embedder = Embedder(dataset.token_dictionary,
#                     dataset.cell2index,
#                     dataset.gene2ensembl)
# embedded_data = embedder.get_embed(
#     model, dataset.test_dataloader(), n_cells=20, include=['gene_emb', 'cell_emb_gene'])
# embs, true_celltype, true_gene = embedded_data.get_all_gene_embedding(
#     'gene_emb')
# cell_embs, cell_celltype = embedded_data['cell_emb_gene'], embedded_data['cell_type']


# encoded_summaries = model.encode_summaries(
#     dataset.get_summaries('gene_celltype', use_names=True), dict_key='cell')
# pred_celltype = classify(embs, encoded_summaries,
#                          mode='cell', return_all=True)
# print(pred_celltype.shape)

# encoded_summaries = model.encode_summaries(
#     dataset.get_summaries('gene_celltype', use_names=True), dict_key='gene')
# pred_celltype = classify(embs, encoded_summaries,
#                          mode='gene', return_all=True)
# print(pred_celltype.shape)

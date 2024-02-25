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

save_dir = './saves'
data_dir = './data'
config = {
    'emb_dim': 1024,
    'freeze_text_model': True,
    'freeze_gene_model': True,
    'use_cell_type': True,
    'lr': 1e-3,
    'batch_size': 16,
    'n_top_genes': 500,
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
    dataset._get_tokenized_gene_sunmmaries(True))

with open('gene_cell_bert_summary.json', 'w') as f:
    json.dump(model.summary_table, f)

# %%
# embedder = Embedder(dataset.token_dictionary,
#                     dataset.cell_index,
#                     dataset.gene2ensembl)

# embedded = embedder.get_embed(model, dataset.val_dataloader(), 100)


# cell_types = np.unique([x['cell_type'] for x in embedded])
# print(cell_types)

# cell_types = cell_types[:2]

# genes = dataset.available_genes[:2]


# save_dir = os.path.join('saves', 'TextGeneContrastive',
# date.today().strftime("%Y-%m-%d"))
# checkpoint_callback = ModelCheckpoint(
# monitor='val_loss',
# dirpath=save_dir,
# filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
# mode='min',
# )
# tblogger = TensorBoardLogger(save_dir)
# trainer = Trainer(max_epochs=20, callbacks=[
#   checkpoint_callback], logger=tblogger)

logger = TensorBoardLogger("temp", name="my_model")
trainer = Trainer(max_epochs=100, logger=logger)
trainer.fit(model, train_dataloaders=dataset.train_dataloader(),
            val_dataloaders=dataset.val_dataloader())


# %%

# %%

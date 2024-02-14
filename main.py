import os
from datetime import date
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import pickle
from LLM4Bio.data import LLM4Bio_data
from LLM4Bio.models import TextGeneContrastive

save_dir = './saves'
config = {
    'emb_dim': 1024,
    'freeze_text_model': True,
    'freeze_gene_model': True,
    'lr': 1e-3,
    'batch_size': 16,
    'n_top_genes': 500,
    'dino_nlayers': 3,
    'data_dir': './data',
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
model.build_summary_table(dataset._get_tokenized_gene_sunmmaries(True))
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

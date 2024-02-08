import os
from datetime import date
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import pickle
from LLM4Bio.data import LLM4Bio_data
from LLM4Bio.models import TextGeneContrastive
dataset = LLM4Bio_data()

dataset.prepare_data()
dataset.setup('')


model = TextGeneContrastive(summary_dict=dataset.get_tokenized_gene_sunmmaries(False),
                            freeze_bert=False,
                            freeze_geneformer=False,
                            lr=1e-3)
model.build_summary_table(dataset.get_tokenized_gene_sunmmaries(True))
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
trainer = Trainer(max_epochs=20)
trainer.fit(model, train_dataloaders=dataset.train_dataloader(),
            val_dataloaders=dataset.val_dataloader())

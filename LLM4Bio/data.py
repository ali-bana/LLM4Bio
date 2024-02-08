from lightning import LightningDataModule
import os
import gdown
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import scanpy as sc
from .Geneformer.tokenizer import TranscriptomeTokenizer
import pickle
from datasets import load_from_disk
from .Geneformer.pretrainer import GeneformerPreCollator
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class LLM4Bio_data(LightningDataModule):
    def __init__(self,
                 data_dir='./data',
                 gene_dataset='PBMC',
                 gene_summary='NCBI',
                 cell_ontology='mixed',  # might neet to remove this one
                 batch_size=8,
                 token_dictionary_path='./LLM4Bio/Geneformer/token_dictionary.pkl',
                 ) -> None:
        super().__init__()
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.data_dir = data_dir
        self.gene_summary = gene_summary
        self.gene_dataset = gene_dataset
        self.cell_ontology = cell_ontology
        self.batch_size = batch_size
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            'michiyasunaga/BioLinkBERT-large')
        with open(token_dictionary_path, 'rb') as f:
            self.token_dictionary = pickle.load(f)

    def prepare_data(self,
                     n_top_genes=500) -> None:
        if self.gene_dataset == 'PBMC':
            adata_url = 'https://drive.google.com/uc?export=download&id=100cUioEzUbO1OTRpNsHwt_Z8XONnVogz'
            self.data_dir = os.path.join(self.data_dir, 'PBMC')
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            adata_path = os.path.join(self.data_dir, 'pbmc_tutorial.h5ad')
        if self.gene_summary == 'NCBI':
            gene_summary_url = 'https://drive.google.com/uc?export=download&id=1snkcCNUyq8IaKFC9sgbe1W76QsE7eHgs'
            gene_summary_path = os.path.join(
                self.data_dir, 'NCBI_gene_summary.txt')
        if self.cell_ontology == 'mixed':
            ontology_url = 'https://drive.google.com/uc?export=download&id=10PVKyXJXEj9_apEAQFcWniBiubk1rOsj'
            cell_ontology_path = os.path.join(
                self.data_dir, 'cell_type_ontology.txt')
        hgnc2ensmbl_url = 'https://drive.google.com/uc?export=download&id=1s9K_tV2f_n6zONtUt6_lTQY_9FYhvZfm'
        hgnc2ensmbl_path = os.path.join(
            self.data_dir, 'hgnc2ensmbl.txt')
        if not os.path.exists(adata_path):
            gdown.download(adata_url, adata_path, quiet=False)
        if not os.path.exists(gene_summary_path):
            gdown.download(gene_summary_url, gene_summary_path)
        if not os.path.exists(cell_ontology_path):
            gdown.download(ontology_url, cell_ontology_path)
        if not os.path.exists(hgnc2ensmbl_path):
            gdown.download(hgnc2ensmbl_url, hgnc2ensmbl_path)
        import json
        with open(hgnc2ensmbl_path, 'r') as f:
            self.gene2ensembl = json.load(f)
        with open(gene_summary_path, 'r') as f:
            self.gene_summary = json.load(f)
        with open(cell_ontology_path, 'r') as f:
            self.ontology = json.load(f)
        print(adata_path)
        adata = sc.read(adata_path)
        loom_path = os.path.join(self.data_dir, 'loom')
        if not os.path.exists(loom_path):
            os.makedirs(loom_path)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata = adata[:, adata.var.highly_variable]
        import pandas as pd
        present_genes = []
        for i in adata.var_names:
            if i in self.gene2ensembl.keys():
                if self.gene2ensembl[i] in self.token_dictionary.keys() and self.gene2ensembl[i] in self.gene_summary.keys():
                    present_genes.append(i)
        adata._inplace_subset_var(present_genes)
        adata.obsm['n_counts'] = adata.X.sum(axis=1)
        adata.varm['ensembl_id'] = pd.Series(
            self.gene2ensembl, index=adata.var_names).values
        self.cell_index = {item: i for i, item in enumerate(
            adata.obs['cell_type'].unique())}
        adata.obs['cell_type'].replace(self.cell_index, inplace=True)
        self.cell_index.update({v: k for k, v in self.cell_index.items()})
        adata.write_loom(os.path.join(loom_path, 'pbmc.loom'), True)
        tk = TranscriptomeTokenizer({"cell_type": "cell_type"}, nproc=16)
        self.tokenized_path = os.path.join(self.data_dir, 'tokenized')
        tk.tokenize_data(loom_path,
                         self.tokenized_path,
                         "",
                         file_format="loom")
        self.tokenized_path = os.path.join(self.data_dir, 'tokenized.dataset')
        self.available_genes = adata.var_names.values
        self.available_genes = [self.gene2ensembl[g]
                                for g in self.available_genes]

    def setup(self, stage: str) -> None:
        dataset = load_from_disk(self.tokenized_path).shuffle(
            seed=42).train_test_split(0.2)
        self.test_dataset = dataset['test']
        dataset = dataset['train'].train_test_split(0.15)
        self.train_dataset = dataset['train']
        self.val_dataset = dataset['test']
        with open("LLM4Bio/Geneformer/token_dictionary.pkl", "rb") as fp:
            self.token_dictionary = pickle.load(fp)
        precollator = GeneformerPreCollator(
            token_dictionary=self.token_dictionary)
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=precollator, mlm=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collator)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collator)

    def get_tokenized_gene_sunmmaries(self, tokenized=True):
        summariers = {}
        for gene in self.available_genes:
            if tokenized:
                summariers[self.token_dictionary[gene]] = self.text_tokenizer(
                    self.gene_summary[gene], return_tensors="pt")
            else:
                summariers[self.token_dictionary[gene]
                           ] = self.gene_summary[gene]
        return summariers

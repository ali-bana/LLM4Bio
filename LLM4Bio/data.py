import anndata as ad
from lightning import LightningDataModule
import os
import gdown
import scanpy as sc
from .Geneformer.tokenizer import TranscriptomeTokenizer
import pickle
from datasets import load_from_disk
from .Geneformer.pretrainer import GeneformerPreCollator
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.auto import tqdm
import numpy as np
from .utils import remove_provided_by, concat_gene_celltype
from transformers import BatchEncoding
from .collator import LLM4BioDataCollator
import json
from .container import EmbeddingContainer
np.random.seed(42)
torch.manual_seed(42)


class LLM4Bio_data(LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        if not os.path.exists(config['data_dir']):
            os.makedirs(config['data_dir'])
        self.data_dir = config['data_dir']
        self.gene_summary = config['gene_summary']
        self.gene_dataset = config['gene_dataset']
        self.cell_ontology = config['cell_ontology']
        self.batch_size = config['batch_size']
        self.freeze_text_model = config['freeze_text_model']
        if 'biolinkbert' in config['text_model'].lower():
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                'michiyasunaga/'+config['text_model'], max_length=512, padding="max_length", truncation=True)
        else:
            self.text_tokenizer = None
        token_dictionary_path = './LLM4Bio/Geneformer/token_dictionary.pkl'
        with open(token_dictionary_path, 'rb') as f:
            self.token_dictionary = pickle.load(f)
        self.n_top_genes = config['n_top_genes']
        self.config = config

    def prepare_data(self) -> None:
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
            # https://drive.google.com/file/d/1K68vSGFxZ_lf5j9CV75l-sj_TW8R5352/view?usp=sharing
            ontology_url = 'https://drive.google.com/uc?export=download&id=1K68vSGFxZ_lf5j9CV75l-sj_TW8R5352'
            cell_ontology_path = os.path.join(
                self.data_dir, 'cell_type_ontology.txt')
        elif self.cell_ontology == 'ChatGPT':
            cell_ontology_path = os.path.join(
                self.data_dir, 'chatgpt_cell_type_summary.json')
        hgnc2ensmbl_url = 'https://drive.google.com/uc?export=download&id=1s9K_tV2f_n6zONtUt6_lTQY_9FYhvZfm'
        hgnc2ensmbl_path = os.path.join(
            self.data_dir, 'hgnc2ensembl.txt')
        kang_path = os.path.join(self.data_dir, 'kang_tutorial.h5ad')
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
            if self.cell_ontology == 'ChatGPT':
                self.ontology = self.ontology['gpt-4']
        # clean gene_summary
        for gene in self.gene_summary.keys():
            self.gene_summary[gene] = remove_provided_by(
                self.gene_summary[gene])
        adata = sc.read_h5ad(adata_path)
        kang = sc.read_h5ad(kang_path)
        kang.obs['study'] = 'Kang'
        test_adata = ad.concat(
            [kang, adata[adata.obs['study'] == 'Oetjen', :].copy()], join='inner')
        adata = adata[adata.obs['study'] != 'Oetjen', :].copy()
        loom_path_train = os.path.join(self.data_dir, 'loom', 'train')
        loom_path_test = os.path.join(self.data_dir, 'loom', 'test')
        if not os.path.exists(loom_path_test):
            os.makedirs(loom_path_test)
        if not os.path.exists(loom_path_train):
            os.makedirs(loom_path_train)
        sc.pp.highly_variable_genes(adata, n_top_genes=self.n_top_genes)
        adata.var.highly_variable = adata.var.highly_variable | adata.var.index.isin(
            self.config['keep_genes'])
        adata = adata[:, adata.var.highly_variable]
        import pandas as pd
        present_genes = []
        for i in adata.var_names:
            if i in self.gene2ensembl.keys():
                if self.gene2ensembl[i] in self.token_dictionary.keys() and self.gene2ensembl[i] in self.gene_summary.keys():
                    present_genes.append(i)
        adata._inplace_subset_var(present_genes)
        test_adata._inplace_subset_var(present_genes)
        adata.obsm['n_counts'] = adata.X.sum(axis=1)
        test_adata.obsm['n_counts'] = test_adata.X.sum(axis=1)
        adata.varm['ensembl_id'] = pd.Series(
            self.gene2ensembl, index=adata.var_names).values
        test_adata.varm['ensembl_id'] = pd.Series(
            self.gene2ensembl, index=test_adata.var_names).values
        self.cell2index = {item: i for i, item in enumerate(
            set(adata.obs['cell_type'].to_list()+test_adata.obs['cell_type'].to_list()))}
        self.study2index = ['10X', 'Freytag', 'Oetjen', 'Sun', 'Kang']
        self.study2index = {s: i for i, s in enumerate(self.study2index)}
        self.index2study = {v: k for k, v in self.study2index.items()}
        train_adata = adata

        print('train adata:\n', train_adata)
        print('test_adata:\n', test_adata)

        train_adata.obs['cell_type'].replace(self.cell2index, inplace=True)
        test_adata.obs['cell_type'].replace(self.cell2index, inplace=True)
        train_adata.obs['study'].replace(self.study2index, inplace=True)
        test_adata.obs['study'].replace(self.study2index, inplace=True)
        self.index2cell = {v: k for k, v in self.cell2index.items()}

        test_adata.write_loom(os.path.join(
            loom_path_test, 'pbmc_test.loom'), True)
        train_adata.write_loom(os.path.join(
            loom_path_train, 'pbmc_train.loom'), True)
        tk = TranscriptomeTokenizer(
            {"cell_type": "cell_type", "study": "study"}, nproc=16)
        self.tokenized_path_train = os.path.join(
            self.data_dir, 'tokenized', 'train')
        self.tokenized_path_test = os.path.join(
            self.data_dir, 'tokenized', 'test')
        tk.tokenize_data(loom_path_train,
                         self.tokenized_path_train,
                         "",
                         file_format="loom")
        tk.tokenize_data(loom_path_test,
                         self.tokenized_path_test,
                         "",
                         file_format="loom")
        self.tokenized_path_train += '.dataset'
        self.tokenized_path_test += '.dataset'
        self.available_genes_train = train_adata.var_names.values
        self.available_genes_train = [self.gene2ensembl[g]
                                      for g in self.available_genes_train]
        self.available_genes_test = test_adata.var_names.values
        self.available_genes_test = [self.gene2ensembl[g]
                                     for g in self.available_genes_test]
        self.available_genes = list(
            set(self.available_genes_train+self.available_genes_test))
        self.available_cells_train = [self.index2cell[i]
                                      for i in train_adata.obs['cell_type'].unique()]
        self.available_cells_test = [self.index2cell[i]
                                     for i in test_adata.obs['cell_type'].unique()]

    def setup(self, stage: str) -> None:
        train_dataset = load_from_disk(self.tokenized_path_train).shuffle(
            seed=42).train_test_split(0.2)
        self.val_dataset = train_dataset['test']
        self.train_dataset = train_dataset['train']
        self.test_dataset = load_from_disk(self.tokenized_path_test)
        with open("LLM4Bio/Geneformer/token_dictionary.pkl", "rb") as fp:
            self.token_dictionary = pickle.load(fp)
        precollator = GeneformerPreCollator(
            token_dictionary=self.token_dictionary)
        gene_collator = DataCollatorForLanguageModeling(
            tokenizer=precollator, mlm=False)

        summaries_gene, summaries_cells = self._get_summaries_for_collator(
            self.config['loss_type'], self.config['use_bert_encoded'], concat_option=self.config['concat_option'])
        self.collator = LLM4BioDataCollator(mode=self.config['loss_type'],
                                            return_encoded=self.config['use_bert_encoded'],
                                            gene_collator=gene_collator,
                                            text_tokenizer=self.text_tokenizer,
                                            summary_dict_gene=summaries_gene,
                                            summary_dict_cell=summaries_cells,
                                            aguments=None)
        # self.collator = DataCollatorForLanguageModeling(
        #     tokenizer=precollator, mlm=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self.collator, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collator, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collator, num_workers=4)

    def get_summaries(self, mode, tokenized=True, use_names=False, concat_option=0):
        if not mode in ['gene_celltype', 'concat_celltype', 'concat', 'gene']:
            raise ValueError(f'mode {mode} is not defined!')
        if 'biolinkbert' in self.config['text_model'].lower():
            result = {}
            gene_summariers = {}
            if 'gene' in mode:
                for gene in self.available_genes:
                    key = self.token_dictionary[gene] if not use_names else gene
                    if tokenized:
                        gene_summariers[key] = self.text_tokenizer(
                            self.gene_summary[gene], return_tensors="pt")
                    else:
                        gene_summariers[key] = self.gene_summary[gene]
                result['gene'] = gene_summariers
            elif 'concat' in mode:
                cells = [idx for idx in self.cell2index.values()]
                for gene in self.available_genes:
                    sentences = []
                    for cell in cells:
                        sentences.append(
                            concat_gene_celltype(gene_string=self.gene_summary[gene],
                                                 cell_string=self.ontology[self.index2cell[cell]],
                                                 gene_name=None,
                                                 cell_name=self.index2cell[cell],
                                                 concat_option=concat_option))
                        key = self.token_dictionary[gene] if not use_names else gene
                        if tokenized:
                            gene_summariers[key] = self.text_tokenizer(
                                sentences, return_tensors="pt", padding=True, max_length=512, truncation=True)
                        else:
                            gene_summariers[key] = sentences
                if use_names:
                    cells = [self.index2cell[idx] for idx in cells]
                result['gene'] = (gene_summariers, cells)
            result['cell'] = {}
            if 'celltype' in mode:
                cell_summaries = {}
                for cell in self.cell2index.keys():
                    key = cell if use_names else self.cell2index[cell]
                    if tokenized:
                        cell_summaries[key] = self.text_tokenizer(
                            self.ontology[cell], return_tensors="pt")
                    else:
                        cell_summaries[key] = self.ontology[cell]
                result['cell'] = cell_summaries
            return result

        elif self.config['text_model'] == 'text-embedding-3-small':
            if 'gene' in mode:
                with open('./data/PBMC/NCBI_gene_summary_embedding.json', 'r') as f:
                    gene_summary_encoded = json.load(
                        f)['text-embedding-3-small']
            elif 'concat' in mode:
                with open('./data/PBMC/concat_embedding.json', 'r') as f:
                    gene_summary_encoded = json.load(
                        f)['text-embedding-3-small']
            with open('data/PBMC/cell_onto_embedding.json', 'r') as f:
                cell_summary_encoded = json.load(f)['text-embedding-3-small']
            result = {'gene': {}, 'cell': {}}
            if 'gene' in mode:
                for gene in self.available_genes:
                    key = self.token_dictionary[gene] if not use_names else gene
                    result['gene'][key] = torch.tensor(
                        gene_summary_encoded[gene], dtype=torch.float32)
            elif 'concat' in mode:
                cells = [idx for idx in self.cell2index.keys()]
                gene_summariers = {}
                for gene in self.available_genes:
                    sentences = []
                    for cell in cells:
                        sentences.append(gene_summary_encoded[gene][cell])
                        key = self.token_dictionary[gene] if not use_names else gene
                        gene_summariers[key] = torch.tensor(
                            sentences, dtype=torch.float32)
                if not use_names:
                    cells = [self.cell2index[c] for c in cells]
                result['gene'] = (gene_summariers, cells)
            result['cell'] = {}
            if 'celltype' in mode:
                for cell in self.cell2index.keys():
                    key = cell if use_names else self.cell2index[cell]
                    result['cell'][key] = torch.tensor(
                        cell_summary_encoded[cell], dtype=torch.float32)
            return result

    def _get_summaries_for_collator(self, mode, encode_using_bert, concat_option=0):

        if self.config['text_model'] == 'text-embedding-3-small' or self.config['text_model'] == 'BioLinkBERT-large':
            embed_size = 1024 if self.config['text_model'] == 'BioLinkBERT-large' else 1536
            container = EmbeddingContainer(
                os.path.join(self.config['text_embedding_dir'], 'embeddings.h5'), embed_size)
            container.open()
            gene_summary = container.get_all_embeddings(
                type='concat' if 'concat' in mode else 'gene')
            cell_summary = container.get_all_embeddings(
                type='cell')
            cell_summary = {self.cell2index[k]: v for k, v in cell_summary.items()}
            gene_summary = {k: v for k, v in gene_summary.items(
            ) if k in self.token_dictionary.keys()}
            if 'concat' in mode:
                gene_summary = {
                    self.token_dictionary[k]: v for k, v in gene_summary.items()}
                for gene in gene_summary.keys():
                    gene_summary[gene] = {
                        self.cell2index[k]: v for k, v in gene_summary[gene].items()}
                gene_summary[0] = {k: np.zeros(embed_size)
                                   for k in self.cell2index.values()}
            elif 'gene' in mode:
                gene_summary = {
                    self.token_dictionary[k]: v for k, v in gene_summary.items()}
                gene_summary[0] = np.zeros(embed_size)
            container.close()
            return gene_summary, cell_summary
        else:
            raise ValueError(
                f"Text model {self.config['text_model']} is not supported!")

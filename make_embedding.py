import scanpy as sc
import json
import pandas as pd
from LLM4Bio.embedding_dataset_maker import make_dataset
import numpy as np
from LLM4Bio.container import EmbeddingContainer
from LLM4Bio.openAI.encode import get_encoding
np.random.seed(0)
with open('data/PBMC/marker_genes_filtered.json', 'r') as f:
    marker_genes = json.load(f)

with open('data/PBMC/NCBI_gene_summary.txt', 'r') as f:
    gene_summaries = json.load(f)

with open('data/PBMC/chatgpt_cell_type_summary.json', 'r') as f:
    cell_summaries = json.load(f)

with open('data/PBMC/hgnc2ensembl.txt', 'r') as f:
    hgnc2ensembl = json.load(f)
ensembl2gene = {v: k for k, v in hgnc2ensembl.items()}

make_dataset(
    dataset_dir='./data',
    dataset_name='openai_te3_embedding_all',
    gene_summaries=gene_summaries,
    cell_summaries=cell_summaries['gpt-4'],
    marker_genes=marker_genes,
    ensembl2gene=ensembl2gene,
    train_test_marker_genes_split=0.3,
    embedding_model='text-embedding-3-small',
)

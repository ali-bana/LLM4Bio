import scanpy as sc
import json
from LLM4Bio.container import EmbeddingContainer
from LLM4Bio.utils import concat_gene_celltype
from tqdm.auto import tqdm
from LLM4Bio.openAI.encode import get_encoding
import pickle
from LLM4Bio.scraper.NCBI_gene_scraper import get_NCBI_summary
with open('./data/openai_te3_embedding_all/using_markers.json', 'r') as f:
    marker_genes = json.load(f)

with open('data/PBMC/NCBI_gene_summary.txt', 'r') as f:
    gene_summaries = json.load(f)

with open('data/PBMC/chatgpt_cell_type_summary.json', 'r') as f:
    cell_summaries = json.load(f)['gpt-4']

with open('data/PBMC/hgnc2ensembl.txt', 'r') as f:
    gene2ensembl = json.load(f)
ens2gene = {v: k for k, v in gene2ensembl.items()}
with open('LLM4Bio/Geneformer/token_dictionary.pkl', 'rb') as f:
    token_dict = pickle.load(f)

token2ens = {v: k for k, v in token_dict.items()}
ensembl2gene = {v: k for k, v in gene2ensembl.items()}

adata = sc.read_h5ad('data/PBMC/pbmc_tutorial.h5ad')
kang = sc.read_h5ad('data/PBMC/kang_tutorial.h5ad')
cells = set(adata.obs['cell_type'].tolist() + kang.obs['cell_type'].tolist())
container = EmbeddingContainer(
    './data/openai_te3_embedding_all/embeddings.h5', 1536)
container.open()
cell_ems = container.get_all_embeddings('cell')
for cell in cells:
    print(cell in cell_ems.keys())


container.close()

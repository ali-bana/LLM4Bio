import json
import seaborn as sns
import numpy as np
import pandas as pd
from LLM4Bio.utils import get_cosines
import matplotlib.pyplot as plt
import scanpy as sc
print('Hello, world!')
with open('./data/PBMC/NCBI_gene_summary_embedding.json', 'r') as f:
    gene_embeds = json.load(f)['text-embedding-3-small']

with open('data/PBMC/cell_onto_embedding.json', 'r') as f:
    cell_embeds = json.load(f)['text-embedding-3-small']
with open('data/PBMC/hgnc2ensembl.txt', 'r') as f:
    hgnc2ensembl = json.load(f)

adata = sc.read_h5ad('data/PBMC/pbmc_tutorial.h5ad')
sc.pp.highly_variable_genes(adata, n_top_genes=500)
adata = adata[:, adata.var.highly_variable]
available_genes = [hgnc2ensembl[gene]
                   for gene in adata.var_names if gene in hgnc2ensembl]
print(adata.var_names)
gene_embeds = {k: v for k, v in gene_embeds.items() if k in available_genes}
print('Number of genes:', len(gene_embeds))
embeddings, cells = [], []

for k, v in cell_embeds.items():
    embeddings.append(v)
    cells.append(k)

cosines = get_cosines(np.array(embeddings))
df = pd.DataFrame(cosines, index=cells, columns=cells)
sns.heatmap(df)
plt.title('Cosine similarity of cell embeddings using OpenAI embedding')
plt.savefig('./figs/heatmap_cell_openai_embedding.png')
plt.show()

embeddings, genes = [], []

for k, v in gene_embeds.items():
    embeddings.append(v)
    genes.append(k)

cosines = get_cosines(np.array(embeddings))
df = pd.DataFrame(cosines, index=genes, columns=genes)
sns.heatmap(df)
plt.title('Cosine similarity of gene embeddings using OpenAI embedding')
plt.savefig('./figs/heatmap_openai_gene_embedding.png')
plt.show()

for i in range(cosines.shape[0]):
    cosines[i, i] = 0

print('Max cosine:', np.max(cosines))
print('Min cosine:', np.min(cosines))
print('Mean cosine:', np.sum(cosines) /
      (cosines.shape[0] * cosines.shape[1] - cosines.shape[0]))

cs = []
for i in range(cosines.shape[0]):
    for j in range(cosines.shape[1]):
        if i == j:
            continue
        else:
            cs.append(cosines[i, j])

print('Mean cosine:', np.mean(cs))
print('Max cosine:', np.max(cs))
print('Min cosine:', np.min(cs))

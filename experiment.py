# %%
from LLM4Bio.utils import get_cosines
from LLM4Bio.embed import Embedder
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scanpy as sc
import numpy as np
import torch
from LLM4Bio.models import TextGeneContrastive
from LLM4Bio.data import LLM4Bio_data
from tqdm import tqdm
ckpt_path = './saves/2024_02_14/2024_02_14/lr_0.001_fe_True_ft_True_contrastive-epoch=48-val_loss=0.22.ckpt'
checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
config = {
    'emb_dim': 1024,
    'freeze_text_model': True,
    'freeze_gene_model': True,
    'lr': 1e-3,
    'batch_size': 16,
    'n_top_genes': 500,
    'dino_nlayers': 3,
    'data_dir': './data',
    'save_dir': './saves',
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
model.load_state_dict(checkpoint['state_dict'])
model.build_summary_table(dataset._get_tokenized_gene_sunmmaries(True))
# %%
embedder = Embedder(dataset.token_dictionary,
                    dataset.cell_index,
                    dataset.gene2ensembl)
# loader = dataset.test_dataloader()
# del dataset
# embedded = embedder.get_embed(model, loader, 7500)
# del model
# # %%
# filtered = embedder.filter(embedded, None, None, 'cell')

# # %%
# adata = sc.AnnData(X=filtered.drop(columns=['cell_type']).values)
# adata.obs['cell_type'] = filtered['cell_type'].values
# sc.pp.neighbors(adata, use_rep='X')
# sc.tl.umap(adata)
# sc.pl.umap(adata, color='cell_type', save='adata.png')
# # %%
# filtered['cell_type']
# # %%

# marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
#                         'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
#                         'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']

# for gene in marker_genes:
#     filtered = embedder.filter(embedded, None, [gene], 'gene')
#     adata = sc.AnnData(X=filtered.drop(columns=['cell_type', 'gene']).values)
#     if adata.shape[0] == 0:
#         continue
#     adata.obs['cell_type'] = filtered['cell_type'].values
#     sc.pp.neighbors(adata, use_rep='X')
#     sc.tl.umap(adata)
#     sc.pl.umap(adata, color='cell_type',
#                title=f'{gene} embeddings', save=f'{gene}.png')

# %%

text_emb_head = embedder.get_all_gene_text_embedding(model)
text_emb_llm = []
for k, v in model.summary_table.items():
    if k == 0:
        continue
    text_emb_llm.append(v.detach().numpy())
text_emb_llm = np.stack(text_emb_llm)

cosine_head = get_cosines(text_emb_head)
cosine_llm = get_cosines(text_emb_llm)

print(
    f'Head: mean {cosine_head.mean()}, max {cosine_head.max()}, min {cosine_head.min()}')
print(
    f'LLM: mean {cosine_llm.mean()}, max {cosine_llm.max()}, min {cosine_llm.min()}')

sns_plot = sns.heatmap(cosine_head)
sns_plot.figure.savefig('./figures/head.png')

sns_plot = sns.heatmap(cosine_llm)
sns_plot.figure.savefig('./figures/llm.png')

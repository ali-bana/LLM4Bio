# %%
from tqdm.auto import tqdm
from LLM4Bio.utils import remove_provided_by
from LLM4Bio.utils import concat_gene_celltype
import tiktoken
from openai import OpenAI
from os import environ
from dotenv import load_dotenv
import json
import scanpy as sc
load_dotenv()
encoder = tiktoken.encoding_for_model('text-embedding-3-small')
client = OpenAI(api_key=environ.get("OPENAI_API_KEY"))

with open('/home/ali/Desktop/Research/Codes/LLM4Bio/data/PBMC/cell_type_ontology.txt', 'r') as f:
    cell_onto = json.load(f)

with open('data/PBMC/NCBI_gene_summary.txt', 'r') as f:
    gene_summaries = json.load(f)
with open('data/PBMC/hgnc2ensembl.txt', 'r') as f:
    hgnc2ensembl = json.load(f)

with open('data/PBMC/concat_embedding.json', 'r') as f:
    concat_embedding = json.load(f)

gene_summaries = {k: remove_provided_by(v) for k, v in gene_summaries.items()}

adata = sc.read_h5ad('data/PBMC/pbmc_tutorial.h5ad')

available_genes = [hgnc2ensembl[g]
                   for g in adata.var_names if g in hgnc2ensembl]
gene_summaries = {k: v for k, v in gene_summaries.items()
                  if k in available_genes}

# %%
to_be_done = [g for g in gene_summaries.keys() if len(
    concat_embedding['text-embedding-3-small'][g]) < 16]
already_done = [g for g in gene_summaries.keys() if len(
    concat_embedding['text-embedding-3-small'][g]) == 16]
# %%
print(len(to_be_done), len(already_done), len(gene_summaries),
      len(available_genes), len(to_be_done)+len(already_done))

# %%


def get_length(text, encoder):
    return len(encoder.encode(text))


# %%

encoding_batch = []
total_string = ''
outer_break = False
for gene in tqdm(to_be_done):
    gene_summary = gene_summaries[gene]
    for cell, cell_summary in cell_onto.items():
        text = concat_gene_celltype(gene_string=gene_summary,
                                    cell_string=cell_summary,
                                    gene_name='',
                                    cell_name=cell,
                                    concat_option=2)
        if get_length(total_string+' '+text, encoder=encoder) < 7800:
            encoding_batch.append((gene, cell, text))
            total_string += '' + text
        else:
            response = client.embeddings.create(
                input=[v for _, __, v in encoding_batch],
                model="text-embedding-3-small"
            )
            for i, item in enumerate(encoding_batch):
                concat_embedding['text-embedding-3-small'][item[0]
                                                           ][item[1]] = response.data[i].embedding
            encoding_batch = []
            total_string = ''
            encoding_batch.append((gene, cell, text))
            total_string += '' + text
            with open('./data/PBMC/concat_embedding.json', 'w') as f:
                json.dump(concat_embedding, f)

if len(encoding_batch) > 0:
    response = client.embeddings.create(
        input=[v for _, __, v in encoding_batch],
        model="text-embedding-3-small"
    )
    for i, item in enumerate(encoding_batch):
        concat_embedding['text-embedding-3-small'][item[0]
                                                   ][item[1]] = response.data[i].embedding
    encoding_batch = []
    total_string = ''

# %% Testin


len(encoding_batch)


# %%
print(len(cell_onto))


# encoded_gene_summary = {'text-embedding-3-small': {}}
# for k, v in tqdm(gene_summary.items()):
#     if get_length(total_string + v, encoder) < 7000:
#         total_string += v
#         encoding_batch.append((k, v))
#     elif get_length(total_string + v, encoder) < 8000:
#         total_string += v
#         encoding_batch.append((k, v))
#         response = client.embeddings.create(
#             input=[v for k, v in encoding_batch],
#             model="text-embedding-3-small"
#         )
#         for i, item in enumerate(encoding_batch):
#             encoded_gene_summary['text-embedding-3-small'][item[0]
#                                                            ] = response.data[i].embedding
#         encoding_batch = []
#         total_string = ''
#     else:
#         response = client.embeddings.create(
#             input=[v for k, v in encoding_batch],
#             model="text-embedding-3-small"
#         )
#         for i, item in enumerate(encoding_batch):
#             encoded_gene_summary['text-embedding-3-small'][item[0]
#                                                            ] = response.data[i].embedding
#         encoding_batch = []
#         total_string = ''
#         total_string += v
#         encoding_batch.append((k, v))

#     response = client.embeddings.create(
#         input=v,
#         model="text-embedding-3-small"
#     )
#     encoded_gene_summary['text-embedding-3-small'][k] = response.data[0].embedding

# #%%
# response = client.embeddings.create(
#     input=[v for k, v in encoding_batch],
#     model="text-embedding-3-small"
# )
# for i, item in enumerate(encoding_batch):
#     encoded_gene_summary['text-embedding-3-small'][item[0]
#                                                     ] = response.data[i].embedding

# # %%
# encoded_gene_summary
# # %%
# with open('data/PBMC/encoded_gene_summary_text-embedding-3-small.json', 'w') as f:
#     json.dump(encoded_gene_summary, f)

# # %%

# %%

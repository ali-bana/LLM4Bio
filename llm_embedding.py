from transformers import AutoTokenizer, AutoModel
import json
from LLM4Bio.openAI.encode import get_encoding
import pandas as pd
from LLM4Bio.container import EmbeddingContainer
from tqdm.auto import tqdm
import random
import numpy as np
from LLM4Bio.utils import get_cosines
import seaborn as sns
import matplotlib.pyplot as plt

# open_ep = EmbeddingContainer(
#     '/home/ali/Desktop/Research/Codes/LLM4Bio/data/openai_te3_embedding_all/embeddings.h5', 1536)
# bert_ep = EmbeddingContainer(
#     'data/biolinkbert_large_embedding_all/embeddings.h5', 1024)
# bert_ep.open()
# open_ep.open()

# for name, container in zip(['openai', 'biolinkbert'], [open_ep, bert_ep]):
#     gene_embeddings = container.get_all_embeddings('gene')
#     keys = list(gene_embeddings.keys())
#     selected = random.choices(keys, k=200)
#     embeddings = np.array([gene_embeddings[k] for k in selected])
#     cosines = get_cosines(embeddings)
#     mean = np.sum(cosines-np.eye(cosines.shape[0])) / \
#         (cosines.shape[0]*cosines.shape[0]-cosines.shape[0])
#     min = np.min(cosines)
#     max = np.max(cosines-np.eye(cosines.shape[0]))
#     print(name, mean, min, max)
#     df = sns.heatmap(cosines)
#     plt.savefig(f'./figures/gene_cosine_similarity_{name}.png')
#     plt.show()


# open_ep.close()
# bert_ep.close()


# %%


with open('data/PBMC/chatgpt_cell_type_summary.json') as f:
    cell_summaries = json.load(f)

cell_summaries = cell_summaries['gpt-4']
tokenizer = AutoTokenizer.from_pretrained(
    'michiyasunaga/BioLinkBERT-large', model_max_length=512)
model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-large')
embedded = {}
for k, v in cell_summaries.items():
    input = tokenizer(v, return_tensors='pt', padding=True,
                      truncation=True, max_length=512)
    output = model(**input)
    embedded[k] = output.last_hidden_state[0, 0, :].detach().numpy()
    print(embedded[k].shape)

# %%
embeddings = np.array([v for v in embedded.values()])
labels = np.array([k for k in embedded.keys()])
print(embeddings.shape)
cosines = get_cosines(embeddings)
# # %%

# %%
mean = np.sum(cosines-np.eye(cosines.shape[0])) / \
    (cosines.shape[0]*cosines.shape[0]-cosines.shape[0])
min = np.min(cosines)
max = np.max(cosines-np.eye(cosines.shape[0]))
# %%

print(mean, min, max)
df = pd.DataFrame(cosines, columns=labels, index=labels)
sns.heatmap(df, vmin=0, vmax=1)


# %%
plt.title('Cosine Similarity of Cell Type Summaries BioLinkBERT')
plt.savefig('./figures/cosine_similarity_biolinkbert.png')
plt.show()
# %%

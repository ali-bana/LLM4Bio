# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from LLM4Bio.utils import get_cosines
from LLM4Bio.openAI.encode import get_encoding
import json
import transformers


with open('data/PBMC/chatgpt_cell_type_summary.json') as f:
    cell_summaries = json.load(f)

cell_summaries = cell_summaries['gpt-4']

embedded = {k: get_encoding(v) for k, v in cell_summaries.items()}

# %%
embeddings = np.array([v[0] for v in embedded.values()])
labels = np.array([k for k in embedded.keys()])
print(embeddings.shape)
cosines = get_cosines(embeddings)
# # %%

# %%
mean = np.mean(cosines-np.eye(cosines.shape[0]))
min = np.min(cosines)
max = np.max(cosines-np.eye(cosines.shape[0]))
# %%

print(mean, min, max)
df = pd.DataFrame(cosines, columns=labels, index=labels)
sns.heatmap(df, vmin=0, vmax=1)


# %%
plt.title('Cosine Similarity of Cell Type Summaries OpenAI')
plt.savefig('./figures/cosine_similarity_openai.png')
plt.show()
# %%

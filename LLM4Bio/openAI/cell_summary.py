# %%
from tqdm.auto import tqdm
from openai import OpenAI
from os import environ
from dotenv import load_dotenv
import json
import scanpy as sc
# %%
load_dotenv()

client = OpenAI(api_key=environ.get("OPENAI_API_KEY"))

adata = sc.read_h5ad('data/PBMC/pbmc_tutorial.h5ad')
kang = sc.read_h5ad('data/PBMC/kang_tutorial.h5ad')
cells = set(kang.obs['cell_type'].to_list() +
            adata.obs['cell_type'].to_list())

result = {'gpt-4': {}, 'gpt-3.5-turbo': {}}
# %%
cells = set(cells)
for cell in tqdm(cells):
    for model in ['gpt-4', 'gpt-3.5-turbo']:
        if cell in result[model].keys():
            continue
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You provide useful biological information to be used to train a model."},
                {"role": "user", "content": f"Give me a summary of {cell} cell type with no more than 400 words."},
            ]
        )
        result[model][cell] = response.choices[0].message.content

# %%
with open('./data/PBMC/chatgpt_cell_type_summary.json', 'w') as f:
    json.dump(result, f)

# %%

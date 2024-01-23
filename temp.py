#%%
import requests
from lxml import etree
from tqdm import tqdm
import json

import re
import json
from tqdm import tqdm
import pandas as pd

ensembl2summary = {}

gene2ensembl = {}
file = '../ensembl2desc.txt'
with open('./hgnc2ensembl.txt', 'r') as f:
    gene2ensembl = json.load(f)
ensembl_ids = [gene2ensembl[g] for g in gene2ensembl]
#%%
id_ = ensembl_ids[0]
id_ = 'ENSG00000179639'
r = requests.get(f'https://ncbi.nlm.nih.gov/gene/?term={id_}')
# %%
tree = etree.HTML(r.text.encode())

# %%
iterator = tree.findall(".//div[@id='summaryDiv']/dl[@id='summaryDl']/")
# %%
for i in iterator:
    print(i.text)
    print('.....')
    print(''.join(i.itertext()))
    print('00000000000000000000000000')


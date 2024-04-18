import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm
import json
import os
import scanpy as sc
import pickle


def get_NCBI_summary(ensmbl_id):
    r = requests.get(f'https://ncbi.nlm.nih.gov/gene/?term={ensmbl_id}')
    tree = etree.HTML(r.text.encode())
    elements = tree.findall(".//div[@id='summaryDiv']/dl[@id='summaryDl']/")
    for i, element in enumerate(elements):
        if ''.join(element.itertext()).lower() == 'summary':
            return ''.join(elements[i+1].itertext())
    return None


if __name__ == '__main__':
    data_dir = './data'
    adata_path = os.path.join(data_dir, 'pbmc_tutorial.h5ad')
    adata = sc.read(adata_path)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    gene2ensembl = {}
    with open('./hgnc2ensembl.txt', 'r') as f:
        gene2ensembl = json.load(f)
    present_genes = []
    for i in adata.var_names:
        if i in gene2ensembl.keys():
            present_genes.append(i)
    adata._inplace_subset_var(present_genes)
    adata.obsm['n_counts'] = adata.X.sum(axis=1)
    adata.varm['ensembl_id'] = pd.Series(
        gene2ensembl, index=adata.var_names).values
    ensembl_ids = pd.Series(gene2ensembl, index=adata.var_names).values
    descs = {}
    des_file = './gene_summary.txt'
    unknown_file = './no_desc_genes.txt'

    for id_ in tqdm(ensembl_ids):
        r = requests.get(f'https://ncbi.nlm.nih.gov/gene/?term={id_}')
        tree = etree.HTML(r.text.encode())
        elements = tree.findall(
            ".//div[@id='summaryDiv']/dl[@id='summaryDl']/")
        for i, element in enumerate(elements):
            if ''.join(element.itertext()).lower() == 'summary':
                descs[id_] = ''.join(elements[i+1].itertext())
                with open(des_file, 'w') as f:
                    json.dump(descs, f)
        if not id_ in descs.keys():
            with open(unknown_file, 'a') as f:
                f.write(id_+'\n')

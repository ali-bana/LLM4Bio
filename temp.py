import json
import re


def remove_provided_by(string):
    return re.sub(r'\[provided by .*?\]', '', string)


with open('data/PBMC/NCBI_gene_summary.txt', 'r') as f:
    gene_summary = json.load(f)


for gene in gene_summary.keys():
    gene_summary[gene] = remove_provided_by(gene_summary[gene])

with open('data/PBMC/NCBI_gene_summary.txt', 'w') as f:
    json.dump(gene_summary, f)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from .utils import has_same_shape


class Embedding:
    def __init__(self, emb_list) -> None:
        self.embedded = emb_list

    def append(self, emb):
        if len(self.embedded) > 0:
            if not list(emb.keys()) == list(self.embedded[0].keys()):
                raise ValueError(
                    'The keys of the new cell are not the same as the previous ones')
        self.embedded.append(emb)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.embedded[key]
        if isinstance(key, str):
            if not key in self.embedded[0].keys():
                raise ValueError(f'{key} is not a valid key')
            values = [c[key] for c in self.embedded]
            if has_same_shape(values):
                return np.array(values)
            else:
                result = []
                for c in values:
                    for v in c:
                        result.append(v)
                return np.array(result)

    def __str__(self) -> str:
        return f'EmbeddedCells with {len(self.embedded)} cells' + '\n' + f'Keys: {list(self.embedded[0].keys())}'

    def get_gene_embedding(self, gene, key='gene_emb'):
        ''' returns two np.arrays, one embedding of the gene in all cells and the cell types'''
        result = []
        cell_types = []
        for cell in self.embedded:
            index, = np.where(cell['input_ids'] == gene)
            if len(index) > 0:
                index = index[0]
                result.append(cell[key][index, :])
                cell_types.append(cell['cell_type'])
        if len(result) == 0:
            raise ValueError(f'Gene {gene} not found in any cell')
        return np.stack(result), np.array(cell_types)

    def get_cell_embedding(self):
        return self['cell_emb_gene'], self['cell_type']

    def get_all_gene_embedding(self, key_embedding_key='gene_emb'):
        ct = []
        gene = []
        embs = {k: [] for k in key_embedding_key} if isinstance(
            key_embedding_key, list) else []
        for c in self.embedded:
            for i, g in enumerate(c['input_ids']):
                ct.append(c['cell_type'])
                gene.append(g)
                if type(key_embedding_key) == list:
                    for k in key_embedding_key:
                        embs[k].append(c[k][i, :])
                else:
                    embs.append(c[key_embedding_key][i, :])
        if type(key_embedding_key) == list:
            for k in key_embedding_key:
                embs[k] = np.array(embs[k])
            return embs, np.array(ct), np.array(gene)
        else:
            return np.array(embs), np.array(ct), np.array(gene)


class Embedder:
    def __init__(self,
                 token_dict: dict,
                 cell_type_dict: dict,
                 hgnc2ensmbl: dict,
                 ) -> None:
        self.token2ensmble = {}
        for k, v in token_dict.items():
            self.token2ensmble[v] = k
        self.id2cell_type = {}
        for k, v in cell_type_dict.items():
            self.id2cell_type[v] = k

        self.hgnc2ensmbl = hgnc2ensmbl
        self.ensmbl2hgnc = {v: k for k, v in hgnc2ensmbl.items()}

    def get_embed(self, model, dataloader, n_cells=-1, include=['gene_emb', 'text_emb', 'geneformer_emb', 'cell_emb_geneformer', 'cell_emb_gene', 'cell_emb_text']):
        result = Embedding([])
        counter = 0
        for batch in tqdm(dataloader):
            with torch.no_grad():
                out = model.forward(batch.to(model.device))
            length = batch['length']
            gene_encs = out['gene_enc']
            text_encs = out['text_enc']
            cell_encs = out['cell_enc']
            for i in range(gene_encs.shape[0]):
                counter += 1
                if n_cells > -1 and counter > n_cells:
                    return result
                cell_dict = {}
                if 'gene_emb' in include:
                    cell_dict['gene_emb'] = gene_encs[i,
                                                      :length[i], :].detach().cpu().numpy()
                if 'text_emb' in include:
                    cell_dict['text_emb'] = text_encs[i,
                                                      :length[i], :].detach().cpu().numpy()
                if 'geneformer_emb' in include:
                    cell_dict['geneformer_emb'] = out['geneformer_encoded'][i,
                                                                            :length[i], :].detach().cpu().numpy()
                if 'cell_emb_geneformer' in include:
                    cell_dict['cell_emb_geneformer'] = out['geneformer_encoded'][i,
                                                                                 :length[i], :].mean(dim=0).detach().cpu().numpy()
                if 'cell_emb_gene' in include:
                    cell_dict['cell_emb_gene'] = cell_encs[i].detach(
                    ).cpu().numpy()
                if 'cell_emb_text' in include:
                    cell_dict['cell_emb_text'] = text_encs[i,
                                                           :length[i], :].mean(dim=0).detach().cpu().numpy()
                cell_dict['cell_type'] = self.id2cell_type[batch['cell_type'][i].item()]
                cell_dict['input_ids'] = batch['input_ids'][i,
                                                            :length[i]].detach().cpu().numpy()
                cell_dict['input_ids'] = np.array(
                    [self.token2ensmble[x] for x in cell_dict['input_ids']])
                result.append(cell_dict)

        return result

    def filter(self, embedded, cell_types=None, genes=None, mode='gene', embedding='gene'):
        if not mode in ['cell', 'gene']:
            raise ValueError('mode must be "cell" or "gene"')
        if not embedding in ['text', 'gene', 'geneformer']:
            raise ValueError('embedding must be "text" or "gene"')
        result_embedding = []
        result_gene = []
        result_cell_type = []

        if mode == 'cellembedded_labels':
            for cell in embedded:
                if cell_types is None or cell['cell_type'] in cell_types:
                    if embedding == 'gene':
                        emb = cell['cell_emb_gene']
                    elif embedding == 'text':
                        emb = cell['cell_emb_text']
                    elif embedding == 'geneformer':
                        emb = cell['cell_emb_geneformer']
                    result_embedding.append(emb)
                    result_cell_type.append(cell['cell_type'])
            result = pd.DataFrame(result_embedding)
            result['cell_type'] = result_cell_type
            return result

        if genes is not None and not 'ENSG' in genes[0]:
            genes = [self.hgnc2ensmbl[g] for g in genes]
        for cell in embedded:
            if cell_types is None or cell['cell_type'] in cell_types:
                for g in genes if not genes is None else cell['input_ids']:
                    index, = np.where(cell['input_ids'] == g)
                    if len(index) > 0:
                        index = index[0]
                        if embedding == 'gene':
                            emb = cell['gene_emb'][index, :]
                        elif embedding == 'text':
                            emb = cell['text_emb'][index, :]
                        elif embedding == 'geneformer':
                            emb = cell['geneformer_emb'][index, :]
                        result_embedding.append(emb)
                        result_gene.append(g)
                        result_cell_type.append(cell['cell_type'])
        result = pd.DataFrame(np.array(result_embedding))
        result['cell_type'] = result_cell_type
        result['gene'] = result_gene
        return result

    def get_all_gene_text_embedding(self, model):
        result = []
        for k, v in model.summary_table.items():
            if k == 0:
                continue
            result.append(
                model.text_encoder.projection_forward(v).detach().cpu().numpy())
        return np.stack(result)

# %%

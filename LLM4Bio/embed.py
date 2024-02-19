import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


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

    def get_embed(self, model, dataloader, n_cells=-1):
        result = []
        counter = 0
        for batch in tqdm(dataloader):
            with torch.no_grad():
                out = model.forward(batch)
            length = batch['length']
            gene_encs = out['gene_enc']
            text_encs = out['text_enc']
            for i in range(gene_encs.shape[0]):
                counter += 1
                if n_cells > -1 and counter > n_cells:
                    return result
                cell_dict = {}
                cell_dict['gene_emb'] = gene_encs[i,
                                                  :length[i], :].detach().numpy()
                cell_dict['text_emb'] = text_encs[i,
                                                  :length[i], :].detach().numpy()
                cell_dict['cell_emb_gene'] = gene_encs[i,
                                                       :length[i], :].mean(dim=0).detach().numpy()
                cell_dict['cell_emb_text'] = text_encs[i,
                                                       :length[i], :].mean(dim=0).detach().numpy()
                cell_dict['cell_type'] = self.id2cell_type[batch['cell_type'][i].item()]
                cell_dict['input_ids'] = batch['input_ids'][i,
                                                            :length[i]].detach().numpy()
                cell_dict['input_ids'] = np.array(
                    [self.token2ensmble[x] for x in cell_dict['input_ids']])
                result.append(cell_dict)

        return result

    def filter(self, embedded, cell_types=None, genes=None, mode='gene', embedding='gene'):
        if not mode in ['cell', 'gene']:
            raise ValueError('mode must be "cell" or "gene"')
        if not embedding in ['text', 'gene']:
            raise ValueError('embedding must be "text" or "gene"')
        result_embedding = []
        result_gene = []
        result_cell_type = []

        if mode == 'cell':
            for cell in embedded:
                if cell_types is None or cell['cell_type'] in cell_types:
                    result_embedding.append(
                        cell['cell_emb_gene'] if embedding == 'gene' else cell['cell_emb_text'])
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
                        result_embedding.append(
                            cell['text_emb'][index, :] if embedding == 'text' else cell['gene_emb'][index, :])
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
                model.text_encoder.projection_forward(v).detach().numpy())
        return np.stack(result)

# %%

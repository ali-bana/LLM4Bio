import torch
import numpy as np
from transformers import AutoTokenizer


# def classify(embedded_data, embedded_labels, mode='gene', max_genes_per_cell=None):
#     if not mode in ['gene', 'cell', 'gene_cell']:
#         raise ValueError('mode must be "gene", "cell" or "gene_cell"')
#     if mode == 'gene_cell':
#         cell_types = []
#         genes = []
#         el = []
#         for ct in embedded_labels:
#             for g in embedded_labels[ct]:
#                 cell_types.append(ct)
#                 genes.append(g)
#                 if isinstance(embedded_labels[ct][g], torch.Tensor):
#                     el.append(embedded_labels[ct][g].detach().cpu().numpy())
#                 else:
#                     el.append(embedded_labels[ct][g])
#         cell_types_labels, genes_labels, embedded_labels = np.array(
#             cell_types), np.array(genes), np.stack(el)

#         true_genes = []
#         true_ct = []
#         pred_gene = []
#         pred_ct = []
#         for cell in embedded_data:
#             gene_embedding = cell['gene_emb']
#             genes = cell['input_ids']
#             if not max_genes_per_cell is None:
#                 indices = np.random.choice(gene_embedding.shape[0], min(
#                     max_genes_per_cell, gene_embedding.shape[0]), replace=False)
#                 gene_embedding = gene_embedding[indices]
#                 genes = genes[indices]
#             cell_type = np.array([cell['cell_type']
#                                  for _ in range(len(genes))])
#             logits = gene_embedding @ embedded_labels.T
#             pred_ct.append(cell_types_labels[np.argmax(logits, axis=1)])
#             pred_gene.append(genes_labels[np.argmax(logits, axis=1)])
#             true_genes.append(genes)
#             true_ct.append(cell_type)
#         return np.concatenate(true_ct), np.concatenate(true_genes), np.concatenate(pred_ct), np.concatenate(pred_gene)

#     else:
#         if isinstance(embedded_labels[list(embedded_labels.keys())[0]], torch.Tensor):
#             embedded_labels = {k: v.detach().cpu().numpy()
#                                for k, v in embedded_labels.items()}
#         l = []
#         el = []
#         for k, v in embedded_labels.items():
#             l.append(k)
#             el.append(v)
#         labels, embedded_labels = np.array(l), np.stack(el)
#         true_classes = []
#         pred_classes = []
#         for cell in embedded_data:
#             cell_embedding = cell['cell_emb_gene'][None]
#             gene_embedding = cell['gene_emb']
#             cell_type = cell['cell_type']
#             genes = cell['input_ids']
#             if not max_genes_per_cell is None:
#                 indices = np.random.choice(gene_embedding.shape[0], min(
#                     max_genes_per_cell, gene_embedding.shape[0]), replace=False)
#                 gene_embedding = gene_embedding[indices]
#                 genes = genes[indices]
#             if mode == 'gene':
#                 true_classes.append(genes)
#                 logits = gene_embedding @ embedded_labels.T
#             else:
#                 true_classes.append([cell_type])
#                 logits = cell_embedding @ embedded_labels.T
#             preds = labels[np.argmax(logits, axis=1)]
#             pred_classes.append(preds)
#         return np.concatenate(true_classes), np.concatenate(pred_classes)


def classify(embeddings, embedded_labels, mode='gene', return_all=False):
    if not mode in ['gene', 'cell', 'concat']:
        raise ValueError('mode must be "gene", "cell" or "gene_cell"')
    if mode == 'concat':
        cell_types = []
        genes = []
        el = []
        for ct in embedded_labels:
            for g in embedded_labels[ct]:
                cell_types.append(ct)
                genes.append(g)
                if isinstance(embedded_labels[ct][g], torch.Tensor):
                    el.append(embedded_labels[ct][g].detach().cpu().numpy())
                else:
                    el.append(embedded_labels[ct][g])
        cell_types_labels, genes_labels, embedded_labels = np.array(
            cell_types), np.array(genes), np.stack(el)

        cell_types_labels, genes_labels, embedded_labels = np.array(
            cell_types), np.array(genes), np.stack(el)

        logits = embeddings @ embedded_labels.T
        if return_all:
            return cell_types_labels[np.argsort(logits, axis=1)], genes_labels[np.argsort(logits, axis=1)]
        return cell_types_labels[np.argmax(logits, axis=1)], genes_labels[np.argmax(logits, axis=1)]

    else:
        if isinstance(embedded_labels[list(embedded_labels.keys())[0]], torch.Tensor):
            embedded_labels = {k: v.detach().cpu().numpy()
                               for k, v in embedded_labels.items()}
        l = []
        el = []
        for k, v in embedded_labels.items():
            l.append(k)
            el.append(v)
        labels, embedded_labels = np.array(l), np.stack(el)
        logits = embeddings @ embedded_labels.T
        if return_all:
            return labels[np.argsort(logits, axis=1)]
        return labels[np.argmax(logits, axis=1)]

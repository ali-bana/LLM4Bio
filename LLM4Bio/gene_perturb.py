import torch
import numpy as np


def perturb(model, data_loader):
    embeddings, cell_types, genes = [], [], []
    count = 0
    for batch in data_loader:
        count += 1
        if count > 10:
            break
        with torch.no_grad():
            idxs = []
            for i in range(batch['input_ids'].shape[0]):
                idx = np.random.choice(batch['length'][i].item())
                genes.append(batch['input_ids'][i, idx].item())
                batch['input_ids'][i, idx] = 0
                idxs.append(idx)

            out = model.forward(batch.to(model.device))
            for i, idx in enumerate(idxs):
                embeddings.append(
                    out['gene_enc'][i, idx].detach().cpu().numpy())
                cell_types.append(batch['cell_type'][i].item())

    return np.stack(embeddings), np.array(cell_types), np.array(genes)

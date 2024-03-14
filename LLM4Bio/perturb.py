import numpy as np
import torch
from tqdm.auto import tqdm


def _in_indices(array, tokens):
    inside = array.clone().apply_(lambda x: x in tokens).nonzero(
        as_tuple=True)[0].tolist()
    return inside


def perturb(model, data_loader, selected_tokens=[]):
    embeddings, cell_types, genes = [], [], []
    count = 0
    for batch in tqdm(data_loader):
        # count += 1
        # if count > 10:
        #     break
        with torch.no_grad():
            idxs = []
            for i in range(batch['input_ids'].shape[0]):
                if len(selected_tokens) == 0:
                    idx = np.random.choice(batch['length'][i].item())
                else:
                    choices = _in_indices(
                        batch['input_ids'][i], selected_tokens)
                    if len(choices) == 0:
                        idx = -1
                    else:
                        idx = np.random.choice(choices)
                if not idx == -1:
                    genes.append(batch['input_ids'][i, idx].item())
                    batch['input_ids'][i, idx] = 1
                idxs.append(idx)
            out = model.forward(batch.to(model.device))
            for i, idx in enumerate(idxs):
                if idx == -1:
                    continue
                embeddings.append(
                    out['gene_enc'][i, idx].detach().cpu().numpy())
                cell_types.append(batch['cell_type'][i].item())

    return np.stack(embeddings), np.array(cell_types), np.array(genes)


if __name__ == '__main__':
    a = [1, 2, 4, 5, 4, 3, 4, 1, 6]
    b = [6, 1]
    # answer = 0, 2, 4, 6, 7
    a = torch.tensor(a)
    print(_in_indices(a, b))

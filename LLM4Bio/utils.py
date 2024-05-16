import torch.nn.functional as F
from scipy.spatial import distance
import numpy as np
import math
import torch
import warnings
import re


def remove_provided_by(string):
    return re.sub(r'\[provided by .*?\]', '', string)


def concat_gene_celltype(gene_string, cell_string, gene_name, cell_name, concat_option=0, tissue_name='', gene_markers=None):
    if concat_option == 0:
        if gene_markers is None:
            raise ValueError(
                'gene_markers shouldn ot be None for concat_option 0')
        return f"Here is a an embedding of gene {gene_name}.{gene_string} Belongs to a cell that come from tissue {tissue_name} and cell type {cell_name}. {cell_string}. and gene markers for this cell type are {', '.join(gene_markers)}."
    elif concat_option == 1:
        raise NotImplementedError('concat_option 1 is not implemented')
    elif concat_option == 2:
        return gene_string + f'Expressed in {cell_name} cell.' + cell_string
    elif concat_option == 3:
        return gene_string + f'This gene is expressed in {cell_name} cell.' + cell_string
    elif concat_option == 4:
        return gene_string + f'Based on information from other genes, this is a {cell_name} cell.' + cell_string


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_cosines(embeddings):
    print(embeddings.shape)
    n = embeddings.shape[0]
    similarities = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarities[i, j] = 1 - distance.cosine(
                embeddings[i], embeddings[j])
    return similarities


def clip(gene_emb, text_emb, temperature=1.0):
    logits = (text_emb @ gene_emb.T) / temperature
    targets = torch.arange(logits.shape[0]).to(logits.device)
    tl = F.cross_entropy(logits, targets)
    gl = F.cross_entropy(logits.T, targets)
    loss = ((tl + gl) / (2.0))
    return {'loss': loss, 'text_loss': tl, 'gene_loss': gl}


def new_clip(gene_emb, text_emb, temperature):
    logits = (text_emb @ gene_emb.T) * torch.exp(temperature)
    targets = torch.arange(logits.shape[0]).to(logits.device)
    tl = F.cross_entropy(logits, targets)
    gl = F.cross_entropy(logits.T, targets)
    loss = ((tl + gl) / (2.0))
    return {'loss': loss, 'text_loss': tl, 'gene_loss': gl}


def has_same_shape(list_array):
    if len(list_array) <= 1:
        return True
    if isinstance(list_array[0], list):
        raise ValueError('list_array should be a list of np.arrays')
    if not isinstance(list_array[0], np.ndarray):
        return True
    shape = list_array[0].shape
    for array in list_array:
        if array.shape != shape:
            return False
    return True


def top_k_accuracy(true_label, sorted_labels, k):
    if len(true_label) != len(sorted_labels):
        raise ValueError(
            'true_label and sorted_labels should have the same length')
    if len(true_label) == 0:
        raise ValueError('true_label and sorted_labels should not be empty')
    if k > sorted_labels.shape[1]:
        raise ValueError(
            'k should be less than or equal to the length of sorted_labels')
    if len(true_label.shape) != 1:
        raise ValueError('true_label should be a 1D array')
    n = true_label.shape[0]
    true_label = true_label[:, None].repeat(k, axis=1)
    return np.sum(np.any(true_label == sorted_labels[:, :k], axis=1)) / n

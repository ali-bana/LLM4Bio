from .container import EmbeddingContainer
import os
import numpy as np
import json
from tqdm.auto import tqdm
from .utils import concat_gene_celltype
import tiktoken
from .openAI.encode import get_encoding


def make_dataset(
    dataset_dir: str,
    dataset_name: str,
    gene_summaries: dict,
    cell_summaries: dict,
    ensembl2gene: dict,
    marker_genes: dict,
    train_test_marker_genes_split: float = 0.3,
    embedding_model: str = 'text-embedding-3-small',
):
    encoder = tiktoken.encoding_for_model('text-embedding-3-small')
    dataset_dir = os.path.join(dataset_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if embedding_model == 'text-embedding-3-small':
        embed_dim = 1536
    elif embedding_model == 'text-embedding-large':
        embed_dim = 3072
    embedding_container = EmbeddingContainer(
        os.path.join(dataset_dir, 'embeddings.h5'), embed_dim)
    with open(os.path.join(dataset_dir, 'all_markers.json'), 'w') as f:
        json.dump(marker_genes, f)

    using_markers = {}
    leave_out_markers = {}

    for cell in marker_genes.keys():
        rands = np.random.rand(len(marker_genes[cell]))
        using_markers[cell] = np.array(marker_genes[cell])[
            rands >= train_test_marker_genes_split].tolist()
        leave_out_markers[cell] = np.array(marker_genes[cell])[
            rands < train_test_marker_genes_split].tolist()

    with open(os.path.join(dataset_dir, 'using_markers.json'), 'w') as f:
        json.dump(using_markers, f)
    with open(os.path.join(dataset_dir, 'test.json'), 'w') as f:
        json.dump(leave_out_markers, f)

    embedding_container.open()

    try:
        encoding_batch = []
        total_string = ''
        for gene in tqdm(gene_summaries.keys()):
            for cell in cell_summaries.keys():
                text = concat_gene_celltype(
                    gene_string=gene_summaries[gene],
                    cell_string=cell_summaries[cell],
                    gene_name=ensembl2gene[gene],
                    cell_name=cell,
                    tissue_name='Primary Peripheral Blood Mononuclear Cells (PBMC)',
                    gene_markers=using_markers[cell],
                    concat_option=0
                )
                if len(encoder.encode(total_string+'\n '+text)) < 7800:
                    encoding_batch.append((gene, cell, text))
                    total_string += ' \n' + text
                else:
                    encs = get_encoding([v for _, __, v in encoding_batch])
                    for i, item in enumerate(encoding_batch):
                        embedding_container.add_embedding(
                            gene=item[0],
                            cell_type=item[1],
                            embedding=encs[i],
                            string=item[2]
                        )
                    encoding_batch = []
                    total_string = ''
                    encoding_batch.append((gene, cell, text))
                    total_string += '' + text
        print('Embedding Finished Successfully')
    except Exception as e:
        print(e)

    embedding_container.close()

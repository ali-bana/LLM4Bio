from LLM4Bio.container import EmbeddingContainer
from LLM4Bio.openAI.encode import get_encoding
import numpy as np

container = EmbeddingContainer('data/temp/PBMC/embeddings.h5', 1536)
container.open()
print(container._cell_types.shape)
for i in range(container._cell_types.shape[0]):
    gene, cell, embedding, string = container.get_idx(i)
    # print(gene, cell, embedding.shape, string)
    enc_new = np.array(get_encoding([string], 'text-embedding-3-small')[0])
    # container.add_embedding(gene, cell, enc_new, string)
    print((embedding-enc_new).max() > 0.0002)
    if (embedding-enc_new).max() > 0.0002:
        print((embedding-enc_new).max())
    print('--------')


container.close()

from LLM4Bio.container import EmbeddingContainer
from tqdm.auto import tqdm

open_ep = EmbeddingContainer(
    '/home/ali/Desktop/Research/Codes/LLM4Bio/data/openai_te3_embedding_all/embeddings.h5', 1536)
bert_ep = EmbeddingContainer(
    'data/biolinkbert_large_embedding_all/embeddings.h5', 1024)
bert_ep.open()
open_ep.open()
print(len(bert_ep._genes), len(open_ep._genes))
open_ep.close()
bert_ep.close()

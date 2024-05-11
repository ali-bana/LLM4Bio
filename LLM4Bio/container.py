import h5py
import numpy as np


class EmbeddingContainer:
    """A class to store embeddings in an HDF5 file.
    """

    def __init__(self, file_path, embedding_dim):
        self.file_path = file_path
        self._file = None
        self._genes = None
        self._cell_types = None
        self._embeddings = None
        self._embedding_dim = embedding_dim

    def open(self):
        self._file = h5py.File(self.file_path, 'a')
        self._genes = self._file.require_dataset(
            'genes', shape=(0,), maxshape=(None,), dtype='S50')
        self._cell_types = self._file.require_dataset(
            'cell_types', shape=(0,), maxshape=(None,), dtype='S50')
        self._embeddings = self._file.require_dataset(
            'embeddings', shape=(0, self._embedding_dim), maxshape=(None, self._embedding_dim), dtype=np.float32)
        self._strings = self._file.require_dataset('strings', shape=(
            0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))

    def close(self):
        self._file.close()

    def add_embedding(self, gene=None, cell_type=None, embedding=None, string=None):
        if gene is None and cell_type is None:
            raise ValueError("Gene or cell type must be provided.")
        if string is None or embedding is None:
            raise ValueError('String and embedding must be provided.')

        gene = '' if gene is None else gene
        cell_type = '' if cell_type is None else cell_type
        index = np.where((self._genes[:] == gene.encode('utf-8')) &
                         (self._cell_types[:] == cell_type.encode('utf-8')))[0]
        if len(index) > 0:
            print(
                f"Warning: Gene-cell type pair ({gene}, {cell_type}) already exists. Overwriting existing values.")
            index = index[0]
            self._embeddings[index] = embedding
            self._strings[index] = string.encode('utf-8')
        else:
            self._genes.resize((self._genes.shape[0] + 1,))
            self._cell_types.resize((self._cell_types.shape[0] + 1,))
            self._embeddings.resize(
                (self._embeddings.shape[0] + 1, self._embeddings.shape[1]))
            self._strings.resize((self._strings.shape[0] + 1,))
            self._genes[-1] = gene.encode('utf-8')
            self._cell_types[-1] = cell_type.encode('utf-8')
            self._embeddings[-1] = embedding
            self._strings[-1] = string.encode('utf-8')

    def get_embedding(self, gene=None, cell_type=None, return_string=False):
        if gene is None and cell_type is None:
            raise ValueError("Gene or cell type must be provided.")
        gene = '' if gene is None else gene
        cell_type = '' if cell_type is None else cell_type
        index = np.where((self._genes[:] == gene.encode('utf-8')) &
                         (self._cell_types[:] == cell_type.encode('utf-8')))[0]

        if len(index) > 0:
            index = index[0]
            if return_string:
                return self._embeddings[index], self._strings[index].decode('utf-8')

            return self._embeddings[index]
        else:
            return None

    def get_all_embeddings(self, type='gene'):
        if not type in ['gene', 'cell', 'concat']:
            raise ValueError(
                "Invalid type. Must be 'gene', 'cell' or 'concat'.")
        embeddings_dict = {}
        for gene, cell_type, embedding in zip(self._genes[:], self._cell_types[:], self._embeddings[:]):
            gene = gene.decode('utf-8')
            cell_type = cell_type.decode('utf-8')
            if type == 'gene':
                if not cell_type == '':
                    continue
                else:
                    embeddings_dict[gene] = embedding
            elif type == 'cell':
                if not gene == '':
                    continue
                else:
                    embeddings_dict[cell_type] = embedding
            elif type == 'concat':
                if gene == '' or cell_type == '':
                    continue
                if gene not in embeddings_dict:
                    embeddings_dict[gene] = {}
                embeddings_dict[gene][cell_type] = embedding
        return embeddings_dict

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_idx(self, idx):
        return self._genes[idx].decode('utf-8'), self._cell_types[idx].decode('utf-8'), self._embeddings[idx], self._strings[idx].decode('utf-8')

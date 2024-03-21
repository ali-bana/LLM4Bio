# %%
import pickle
import numpy as np
from sklearn import svm

with open('./data/svm_embedding.pkl', 'rb') as file:
    save_embeds = pickle.load(file)
# %%
test_gene_embs = save_embeds['test_gene_embs']
test_celltype = save_embeds['test_celltype']
test_genes = save_embeds['test_genes']
test_cell_embs = save_embeds['test_cell_embs']
test_cell_celltypes = save_embeds['test_cell_celltypes']
test_geneformer_cell_embs = save_embeds['test_geneformer_cell_embs']
test_geneformer_gene_embs = save_embeds['test_geneformer_gene_embs']


train_gene_embs = save_embeds['train_gene_embs']
train_celltype = save_embeds['train_celltype']
train_genes = save_embeds['train_genes']
train_cell_embs = save_embeds['train_cell_embs']
train_cell_celltypes = save_embeds['train_cell_celltypes']
train_geneformer_cell_embs = save_embeds['train_geneformer_cell_embs']
train_geneformer_gene_embs = save_embeds['train_geneformer_gene_embs']

# %%
# predict cell from cell embedding our model
simple_clf = svm.LinearSVC(dual='auto')
simple_clf.fit(train_cell_embs, train_cell_celltypes)
pred_cells = simple_clf.predict(test_cell_embs)
print('using cell embedding from our model, accuracy to predict celltype from cell embedding is',
      (pred_cells == test_cell_celltypes).sum()/test_cell_celltypes.shape[0]
      )
# %%
# predict cell from cell embedding geneformer
simple_clf = svm.LinearSVC(dual='auto')
simple_clf.fit(train_geneformer_cell_embs, train_cell_celltypes)
pred_cells = simple_clf.predict(test_geneformer_cell_embs)
print('using cell embedding from geneformer, accuracy to predict celltype from cell embedding is',
      (pred_cells == test_cell_celltypes).sum()/test_cell_celltypes.shape[0]
      )

# %%
# predict cell from gene embedding our model
simple_clf = svm.LinearSVC(dual='auto')
idxs = np.random.choice(train_gene_embs.shape[0], 15000, replace=False)
simple_clf.fit(train_gene_embs[idxs], train_celltype[idxs])
print('done')
pred_genes = simple_clf.predict(test_gene_embs)
print('using gene embedding from our model, accuracy to predict cell from gene embedding is',
      (pred_genes == test_celltype).sum()/test_celltype.shape[0]
      )
# %%
# predict cell from gene embedding geneformer
simple_clf = svm.LinearSVC(dual='auto')
simple_clf.fit(train_geneformer_gene_embs[idxs], train_celltype[idxs])
print('done')
pred_genes = simple_clf.predict(test_geneformer_gene_embs)
print('using gene embedding from geneformer, accuracy to predict cell from gene embedding is',
      (pred_genes == test_celltype).sum()/test_celltype.shape[0]
      )

# %%
# predict gene from gene embedding our model
simple_clf = svm.LinearSVC(dual='auto')
idxs = np.random.choice(train_gene_embs.shape[0], 15000, replace=False)
simple_clf.fit(train_gene_embs[idxs], train_genes[idxs])
print('done')
pred_genes = simple_clf.predict(test_gene_embs)
print('using gene embedding from our model, accuracy to predict gene from gene embedding is',
      (pred_genes == test_genes).sum()/test_genes.shape[0]
      )
# %%
# Predict gene from geneformer gene embedding
simple_clf = svm.LinearSVC(dual='auto')
simple_clf.fit(train_geneformer_gene_embs[idxs], train_genes[idxs])
print('done')
pred_genes = simple_clf.predict(test_geneformer_gene_embs)
print('using gene embedding from  geneformer, accuracy to predict gene from gene embedding is',
      (pred_genes == test_genes).sum()/test_genes.shape[0]
      )
# %%

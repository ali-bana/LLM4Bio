from transformers import AutoTokenizer, AutoModel
from LLM4Bio import TextEncoder
from LLM4Bio.Geneformer import TranscriptomeTokenizer
import anndata as ad
import gdown
import scanpy as sc
import os
cell_type_desc = {
    'NKT cells':'A mature alpha-beta T cell of a distinct lineage that bears natural killer markers and a T cell receptor specific for a limited set of ligands. NK T cells have activation and regulatory roles particularly early in an immune response.',
    'NK cells':"NK cells, vital to innate immunity, swiftly target virus-infected and stressed cells, providing rapid responses. Recognized by CD56+ and absence of CD3, they differentiate from CD127+ progenitors, maturing in various tissues. CD16 and CD57 surface markers distinguish them. Apart from innate functions, NK cells contribute to adaptive immunity, demonstrating antigen-specific memory. Their pivotal role in cancer and HIV therapy underscores their therapeutic potential.",
    'CD4+ T cells':'CD4+ T cells, crucial immune orchestrators, arise in the thymus and regulate immune responses. Recognizing antigens, they coordinate immune defenses by activating other immune cells. Integral in adaptive immunity, CD4+ T cells play key roles in infection control and immune memory.',
    'CD14+ Monocytes':'CD14+ cells, predominantly monocytes and macrophages, express CD14 surface protein. Key players in innate immunity, they detect bacterial components, initiating immune responses. CD14+ cells contribute to inflammation, phagocytosis, and immune regulation, essential for host defense.',
    'HSPCs':"Hematopoietic Stem and Progenitor Cells (HSPCs) are the wellspring of blood cells, residing in the bone marrow. With self-renewal capability, they give rise to diverse blood cell types, playing a fundamental role in maintaining the body's immune and oxygen transport systems.",
    'CD16+ Monocytes':'CD16+ Monocytes, a subset of white blood cells, express CD16 surface marker. Versatile in immune responses, they participate in inflammation, pathogen recognition, and antibody-dependent cellular cytotoxicity. CD16+ Monocytes contribute to host defense and immune regulation.',
    'CD8+ T cells':'CD8+ T cells, originating in the thymus, express CD8 co-receptor and recognize antigens presented by MHC Class I. Vital for defense against intracellular pathogens and tumors, they employ cytokine secretion, cytotoxic granule release, and Fas/FasL interactions to induce apoptosis in infected or malignant cells. These cells exhibit precise targeting to avoid bystander damage and can engage in serial killing. However, their actions may also contribute to immunopathology, highlighting their dual role in immune responses.',
    'CD10+ B cells':"CD10+ B cells express the 100 kD cell surface glycoprotein CD10, a peptidase M13 family member. Initially identified in acute lymphoblastic leukemia, CD10 serves as a cell surface marker for hematological malignancies. Its expression, implicated in cancer, is linked to angiogenesis and exhibits diverse roles—acting as a tumor suppressor in some cancers and promoting tumorigenesis in others. CD10's regulatory functions and structural features contribute to its significance in cancer development and progression.",
    'Plasmacytoid dendritic cells':"Plasmacytoid dendritic cells (pDCs), rare immune cells (< 0.4 precent of PBMC), secrete type 1 interferon in viral responses, bridging innate and adaptive immunity. While crucial in antiviral defense, pDCs' involvement in autoimmune diseases, including lupus, and malignant transformation leading to blastic plasmacytoid dendritic cell neoplasm are noted.",
    'Monocyte progenitors':"Monocyte progenitor cells are precursors in hematopoiesis, arising from hematopoietic stem cells. They give rise to monocytes, essential components of the immune system. Monocyte progenitors differentiate in the bone marrow and circulate in the bloodstream, contributing to immune surveillance and inflammation upon activation.",
    'B':'A lymphocyte of B lineage that is capable of B cell mediated immunity.',
    'Monocyte-derived dendritic cells':'A dendritic cell that develops from a monocyte.',
    'Plasma cells':'A terminally differentiated, post-mitotic, antibody secreting cell of the B cell lineage with the phenotype CD138-positive, surface immunonoglobulin-negative, and MHC Class II-negative. Plasma cells are oval or round with extensive rough endoplasmic reticulum, a well-developed Golgi apparatus, and a round nucleus having a characteristic cartwheel heterochromatin pattern and are devoted to producing large amounts of immunoglobulin. Plasma cells develop in the spleen and migrate to the bone marrow. Plasma cells are also reportedly CD5-negative, CD10-negative, CD19-positive, CD20-negative, CD21-negative, CD22-negative, CD23-negative, CD24-negative, CD25-negative, CD27-positive, CD34-negative, CD38-positive, CD40-positive, CD43-positive, CD45-positive, CD48-positive, CD53-low, CD80-negative, CD81-positive, CD86-positive, CD95-positive, CD196-negative, CD229-positive, CD270-positive, CD352-positive, CD361-positive, and IgD-negative. Transcription factors: BLIMP1-positive, IRF4-positive, PAX5-negative, SpiB-negative, Ets1-negative, and XBP1-positive.',
    'Erythroid progenitors':'A progenitor cell committed to the erythroid lineage.',
    'Megakaryocyte progenitors':'The earliest cytologically identifiable precursor in the thrombocytic series. This cell is capable of endomitosis and lacks expression of hematopoieitic lineage markers (lin-negative). Lineage negative is described here as CD2-negative, CD3-negative, CD4-negative, CD5-negative, CD8a-negative, CD14-negative, CD19-negative, CD20-negative, CD56-negative, Ly6g-negative, and Ter119-negative.'   
}
data_dir = './data'
url = 'https://drive.google.com/uc?id=1Rnm-XKEqPLdOq3lpa3ka2aV4bOXVCLP0'
adata_path = os.path.join(data_dir, 'pbmc_tutorial.h5ad')
gdown.download(url, adata_path, quiet=False)
adata = sc.read(adata_path)

loom_path = './data/loom'
if not os.path.exists(loom_path): os.makedirs(loom_path)
print('before removing non GPs!')
print(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=500)
adata = adata[:, adata.var.highly_variable]
print(adata)
print('Removing genes not in Ensembl')
import json
gene2ensembl = {}
with open('./hgnc2ensembl.txt', 'r') as f:
    gene2ensembl = json.load(f)
import pandas as pd
present_genes = []
for i in adata.var_names:
       if i in gene2ensembl.keys():
              present_genes.append(i)
adata._inplace_subset_var(present_genes)
adata.obsm['n_counts'] = adata.X.sum(axis=1)
adata.varm['ensembl_id'] = pd.Series(gene2ensembl, index=adata.var_names).values
print(adata)
adata.write_loom(os.path.join(loom_path, 'pbmc.loom'), True)
tk = TranscriptomeTokenizer({"cell_type": "cell_type"}, nproc=16)
tk.tokenize_data(loom_path, 
                 "tekenized", 
                 "", 
                 file_format="loom")
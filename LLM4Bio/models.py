from lightning import LightningModule
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel




class TextEncoder(nn.Module):
    def __init__(self, 
                 emb_dim = 1024,
                 llm_model = 'bioLinkBert',
                 freeze_llm = True,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if llm_model.lower() == 'biolinkbert':
            self.model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-base')
            llm_emb_dim = 768
        if freeze_llm:
            for param in self.model.parameters():
                param.requires_grad = False
        self.projector = ProjectionHead(embedding_dim=llm_emb_dim,
                                        projection_dim=emb_dim,
                                        dropout=0.2)
        
    def forward(self, inputs):
        out = self.model(**inputs)
        out = out.last_hidden_state
        out = self.projector.forward(out.mean(axis=1))
        return out

        







class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()


        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)


        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)


    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)


        x += projected


        return self.layer_norm(x)


from typing import Any, Optional
from pytorch_lightning import LightningModule
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .utils import trunc_normal_
from tqdm import tqdm
import torch
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        if config['text_model'].lower() == 'biolinkbert':
            self.model = AutoModel.from_pretrained(
                'michiyasunaga/BioLinkBERT-base')
            llm_emb_dim = 768
        if config['freeze_text_model']:
            for param in self.model.parameters():
                param.requires_grad = False
        self.projector = DINOHead(
            llm_emb_dim, config['emb_dim'], nlayers=config['dino_nlayers'])

    def forward(self, inputs):
        return self.projection_forward(self.llm_forward(inputs))

    def llm_forward(self, inputs):
        return self.model(**inputs).last_hidden_state

    def projection_forward(self, inputs):
        return self.projector.forward(inputs)


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


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class GeneEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(
            "ctheodoris/Geneformer")
        if config['freeze_gene_model']:
            for param in self.model.parameters():
                param.requires_grad = False
        geneformer_dim = 256
        self.projector = DINOHead(
            geneformer_dim, config['emb_dim'], nlayers=config['dino_nlayers'])

    def gene_encoder_forward(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        return self.model.forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  output_hidden_states=True).hidden_states[-1]

    def projection_forward(self, inputs):
        return self.projector(inputs)

    def forward(self, inputs):
        return self.projection_forward(self.gene_encoder_forward(inputs))


class TextGeneContrastive(LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_dim = config['emb_dim']
        self.text_encoder = TextEncoder(config)
        self.gene_encoder = GeneEncoder(config)
        self.summary_table = dict()
        self.temperature = 1
        self.lr = config['lr']
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            'michiyasunaga/BioLinkBERT-large')  # add this to dataloader
        self.freeze_bert = config['freeze_text_model']

    def build_summary_table(self, tokenized_gene_summary: dict):
        if not self.freeze_bert:
            return
        with torch.no_grad():
            for gene in tqdm(tokenized_gene_summary.keys(), desc="Building summary table"):
                self.summary_table[gene] = self.text_encoder.llm_forward(
                    tokenized_gene_summary[gene])[0][0]
            self.summary_table[0] = torch.zeros(768)

    def forward(self, inputs) -> Any:
        gene_enc = self.gene_encoder.forward(inputs)
        input_ids = inputs['input_ids']
        text_enc = []
        if self.freeze_bert:
            for i in range(input_ids.shape[0]):
                emb = []
                for j in range(input_ids.shape[1]):
                    emb.append(self.summary_table[input_ids[i, j].item()])
                text_enc.append(torch.stack(emb))
        else:
            for i in range(input_ids.shape[0]):
                ins = [self.summary_dict[int(input_ids[i][j])]
                       for j in range(input_ids.shape[1])]
                ins = self.text_tokenizer(
                    ins, padding=True, return_tensors='pt').to(gene_enc.device)
                text_enc.append(self.text_encoder.llm_forward(ins).mean(dim=1))
        text_enc = torch.stack(text_enc).to(gene_enc.device)
        text_enc = self.text_encoder.projection_forward(
            text_enc)

        return {'gene_enc': gene_enc, 'text_enc': text_enc}

    def loss(self, inputs, outputs):
        gene_loss = 0
        text_loss = 0
        loss = 0
        length = inputs['length']
        for i in range(inputs['input_ids'].shape[0]):
            text_embedding = outputs['text_enc'][i, :length[i]]
            gene_embedding = outputs['gene_enc'][i, :length[i]]
            logits = (text_embedding @ gene_embedding.T) / self.temperature
            targets = torch.arange(logits.shape[0]).to(logits.device)
            tl = F.cross_entropy(logits, targets)
            gl = F.cross_entropy(logits.T, targets)
            loss += ((tl + gl) / (2.0))
            text_loss += tl
            gene_loss += gl
        return {'gene_loss': gene_loss, 'text_loss': text_loss, 'loss': loss}

    def _clip_with_logits(self, logits):
        n = logits.shape[1]      # number of samples
        labels = torch.arange(n)  # Create labels tensor
        # Calculate cross entropy losses along axis 0 and 1
        loss_i = F.cross_entropy(logits.transpose(
            0, 1), labels, reduction="mean")
        loss_t = F.cross_entropy(logits, labels, reduction="mean")
        # Calculate the final loss
        loss = (loss_i + loss_t) / 2
        return loss

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss_dict = self.loss(batch, outputs)
        logs = {
            'loss': loss_dict['loss'],
            'train_loss': loss_dict['loss'],
            'train_gene_loss': loss_dict['gene_loss'],
            'train_text_loss': loss_dict['text_loss']
        }
        self.log_dict(logs)
        return loss_dict

    def validation_step(self, batch, batch_idx: int):
        out_puts = self.forward(batch)
        loss_dict = self.loss(batch, out_puts)
        logs = {
            'val_loss': loss_dict['loss'],
            'val_gene_loss': loss_dict['gene_loss'],
            'val_text_loss': loss_dict['text_loss']
        }
        self.log_dict(logs)
        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

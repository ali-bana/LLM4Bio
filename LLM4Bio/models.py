from typing import Any, Optional
from pytorch_lightning import LightningModule
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .utils import trunc_normal_
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .utils import clip


class TextEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        if 'biolinkbert' in config['text_model'].lower():
            self.model = AutoModel.from_pretrained(
                'michiyasunaga/'+config['text_model'])
            llm_emb_dim = 768
        if config['freeze_text_model']:
            for param in self.model.parameters():
                param.requires_grad = False
        self.projector = DINOHead(
            llm_emb_dim, config['emb_dim'], nlayers=config['dino_nlayers'])

    def forward(self, inputs):
        return self.projection_forward(self.llm_forward(inputs)[:, 0, :])

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
        self.save_hyperparameters()
        self.embed_dim = config['emb_dim']
        self.text_encoder = TextEncoder(config)
        self.gene_encoder = GeneEncoder(config)
        self.temperature = config['temperature']
        self.lr = config['lr']
        self.mode = config['loss_type']
        self.encoded_input = config['use_bert_encoded']
        self.config = config

    def encode_summaries(self, tokenized_summaries, dict_key='gene'):
        result = {}
        tokenized_summaries = tokenized_summaries[dict_key]
        if isinstance(tokenized_summaries, dict):
            with torch.no_grad():
                for key in tokenized_summaries.keys():
                    result[key] = self.text_encoder.forward(
                        tokenized_summaries[key].to(self.text_encoder.model.device))[0][0].detach().cpu().numpy()

        else:
            with torch.no_grad():
                summaries, cells = tokenized_summaries
                for cell in cells:
                    result[cell] = dict()
                for gene in tqdm(summaries.keys(), desc="Encoding Sumaries"):
                    out = self.text_encoder.forward(
                        summaries[gene].to(self.text_encoder.model.device))
                    for i, cell in enumerate(cells):
                        result[cell][gene] = out[i][0].detach().cpu().numpy()
                    del out
        return result

    def forward(self, inputs) -> Any:
        inputs_gene = inputs['gene']
        inputs_text = inputs['text']
        batch_size = inputs_gene['input_ids'].shape[0]
        geneformer_encoding = self.gene_encoder.gene_encoder_forward(
            inputs_gene)
        gene_enc = self.gene_encoder.projection_forward(geneformer_encoding)
        cell_enc = (gene_enc *
                    inputs_gene['attention_mask'][:, :, None]).sum(dim=1)
        cell_enc = cell_enc / inputs_gene['length'][:, None]
        if self.encoded_input:
            gene_text_enc = self.text_encoder.projection_forward(
                inputs_text['gene_bert_encoded'])
            cell_text_enc = self.text_encoder.projection_forward(
                inputs_text['cell_bert_encoded'])
        else:
            cell_text_enc = self.text_encoder.forward(
                inputs_text['cell_summary'])
            gene_text_enc = []
            max_length = inputs_gene['length'].max().item()
            # in order to prevent cuda out of memory error, we need to iterate over each cell for data
            for cell_data in inputs_text['gene_summary']:
                # if cell_data['input_ids'].shape[0] <= batch_size: # uncomment this line if you want to use this part
                if True:

                    gene_text_enc.append(
                        nn.functional.pad(self.text_encoder.forward(cell_data), (0, 0, 0, max_length - cell_data['input_ids'].shape[0]), value=0))
                else:
                    # still, if the dat
                    raise NotImplementedError(
                        'This part will be used in case colab cannot handle big data block. TB implemented.')
            gene_text_enc = torch.stack(gene_text_enc, dim=0)
            cell_text_enc = (gene_text_enc *
                             inputs_gene['attention_mask'][:, :, None]).sum(dim=1)
            cell_text_enc = cell_text_enc / inputs_gene['length'][:, None]
        return {'gene_enc': gene_enc,
                'cell_enc': cell_enc,
                'text_enc': gene_text_enc,
                'cell_text_enc': cell_text_enc,
                'geneformer_encoded': geneformer_encoding
                }

    def loss(self, inputs, outputs):
        gene_loss = 0
        text_loss = 0
        loss = 0
        inputs = inputs['gene']
        length = inputs['length']
        for i in range(inputs['input_ids'].shape[0]):
            text_embedding = outputs['text_enc'][i, :length[i]]
            gene_embedding = outputs['gene_enc'][i, :length[i]]
            cell_loss = clip(gene_embedding, text_embedding, self.temperature)
            loss += cell_loss['loss']
            text_loss += cell_loss['text_loss']
            gene_loss += cell_loss['gene_loss']
        if 'celltype' in self.mode:
            cell_loss = clip(outputs['cell_enc'],
                             outputs['cell_text_enc'], self.temperature)
            loss += cell_loss['loss']
            text_loss += cell_loss['text_loss']
            gene_loss += cell_loss['gene_loss']
        return {'gene_loss': gene_loss, 'text_loss': text_loss, 'loss': loss}

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

import numpy as np
import torch
from transformers import BatchEncoding


class ListInputHolder:
    def __init__(self, input_list):
        self.input_list = input_list

    def __getitem__(self, idx):
        return self.input_list[idx]

    def __len__(self):
        return len(self.input_list)

    def __str__(self) -> str:
        return str(self.input_list)

    def __repr__(self) -> str:
        return self.__str__()

    def to(self, device):
        self.input_list = [x.to(device) for x in self.input_list]
        return self


class LLM4BioDataCollator:
    def __init__(self,
                 mode,
                 return_encoded,
                 gene_collator,
                 text_tokenizer,
                 summary_dict_gene,
                 summary_dict_cell,
                 aguments=None
                 ):
        """Data collator for multimodal gene and text description of the data

        Args:
            mode (str): it is the mode of the loss function.
            return_encoded (bool): If the model returns the encoded text (using bert) or the tokenized text.
            gene_collator : collator function for geneformer
            text_tokenizer : tokenizer we will be using for text
            summary_dictaguments (list, optional): Agumentation functions to be used on the text data. Defaults to None.
        """
        self.gene_collator = gene_collator
        self.text_tokenizer = text_tokenizer
        self.summary_dict_gene, self.summary_dict_cell = summary_dict_gene, summary_dict_cell
        self.return_encoded = return_encoded
        self.mode = mode
        self.agumentations = aguments
        self.agument = False if aguments is None or len(
            aguments) == 0 else True

    def __call__(self, batch):
        """Given a batch of data, it collates the data into a single batch
        If return_encoded is True, it returns the encoded text using bert, otherwise it returns the tokenized text.
        collated_batch will have a structure as follows:
        'gene' = gene data returned by geneformer collator.
        if case of return_encoded==False:
            'text' = {'cell_summary': tensor, 'gene_summary': ListInputHolder(BatchEncoding)}
        otherwise:
            'text' = {'gene_bert_encoded': tensor, 'cell_bert_encoded': tensor}

        Args:
            batch (list): list of python dictionaries containing the data

        Returns:
            collated_batch (BatchEncoding): collated batch of data
        """
        collated_batch = BatchEncoding()
        collated_batch['gene'] = self.gene_collator(batch)
        collated_batch['text'] = BatchEncoding()
        gene_summaries = []
        cell_summaries = []

        if self.return_encoded:
            for gene_seq, celltype in zip(collated_batch['gene']['input_ids'], collated_batch['gene']['cell_type']):
                if 'gene' in self.mode:
                    gene_summaries.append(
                        [self.summary_dict_gene[gene.item()] for gene in gene_seq])
                elif 'concat' in self.mode:
                    gene_summaries.append(
                        [self.summary_dict_gene[gene.item()][celltype.item()] for gene in gene_seq])
                cell_summaries.append(self.summary_dict_cell[celltype.item()])
            collated_batch['text']['gene_bert_encoded'] = torch.tensor(
                gene_summaries, device=collated_batch['gene']['input_ids'].device, dtype=torch.float32)
            collated_batch['text']['cell_bert_encoded'] = torch.tensor(
                cell_summaries, device=collated_batch['gene']['input_ids'].device, dtype=torch.float32)
        else:
            for gene_seq, celltype in zip(collated_batch['gene']['input_ids'], collated_batch['gene']['cell_type']):
                gs = []
                for gene in gene_seq:
                    if gene.item() == 0:
                        break
                    if 'gene' in self.mode:
                        gs.append(self.summary_dict_gene[gene.item()])
                    elif 'concat' in self.mode:
                        gs.append(
                            self.summary_dict_gene[gene.item()][celltype.item()])
                cell_summaries.append(self.summary_dict_cell[celltype.item()])
                gene_summaries.append(gs)
            cell_summaries = self.text_tokenizer(
                cell_summaries, padding=True, truncation=False, return_tensors='pt')
            collated_batch['text']['cell_summary'] = cell_summaries

            gene_summaries = ListInputHolder([self.text_tokenizer(
                seqs, padding=True, truncation=False, return_tensors='pt') for seqs in gene_summaries])
            collated_batch['text']['gene_summary'] = gene_summaries

        return collated_batch

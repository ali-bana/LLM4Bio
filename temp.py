# %%
import datasets
from LLM4Bio.Geneformer.pretrainer import GeneformerPreCollator
import pickle
from transformers import DataCollator, DataCollatorForLanguageModeling
from datetime import date


print(date.today().strftime("%Y_%m_%d"))

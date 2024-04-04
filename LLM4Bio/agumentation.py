# %%
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data
import json
nltk.download('universal_tagset')


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


with open('/home/ali/Desktop/Research/Codes/LLM4Bio/data/PBMC/cell_type_ontology.txt', 'r') as f:
    cell_type_ontology = json.load(f)

# %%


def agument(text, p=0.3):

    tokenized = [word_tokenize(t) for t in nltk.tokenize.sent_tokenize(text)]
    result = ''
    pos_tagged = nltk.pos_tag_sents(tokenized)
    for sentence in pos_tagged:
        for word, treebank_tag in sentence:
            pos = get_wordnet_pos(treebank_tag)
            print(word, pos)

    return result


text = list(cell_type_ontology.values())[0]

print(text)
print(agument(text))

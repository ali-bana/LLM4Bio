from openai import OpenAI
from os import environ
from dotenv import load_dotenv
import json
import scanpy as sc
load_dotenv()
client = OpenAI(api_key=environ.get("OPENAI_API_KEY"))


def get_encoding(text, model: str = 'text-embedding-3-small') -> list:
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return [d.embedding for d in response.data]

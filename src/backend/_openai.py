import numpy as np
from tqdm import tqdm
from typing import Union, List

from openai import AzureOpenAI

from src.backend import BaseEmbedder


class OpenAIBackend(BaseEmbedder):
    """Flair Embedding Model
    The Flair embedding model used for generating document and
    word embeddings.

    Arguments:
        embedding_model: A OpenAI embedding model

    """

    def __init__(self, embedding_model: AzureOpenAI):
        super().__init__()

        # Flair word embeddings
        if isinstance(embedding_model, AzureOpenAI):
            self.embedding_model = embedding_model

        else:
            raise ValueError(
                "Error while connecting to OpenAI client."
            )

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional
        matrix of embeddings
        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process
        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        embeddings = []
        for document in tqdm(documents):
            embedding = self.embedding_model.embeddings.create(
                input=[document],
                model='text-embedding-3-large'
            ).data[0].embedding
            embedding = np.array(embedding)
            embeddings.append(embedding)
        embeddings = np.asarray(embeddings)
        return embeddings

from ._base import BaseEmbedder


def select_backend(embedding_model) -> BaseEmbedder:
    """Select an embedding model based on language or a specific sentence transformer models.
    When selecting a language, we choose `all-MiniLM-L6-v2` for English and
    `paraphrase-multilingual-MiniLM-L12-v2` for all other languages as it support 100+ languages.

    Returns:
        model: Either a Sentence-Transformer or Flair model
    """
    # keybert language backend
    if isinstance(embedding_model, BaseEmbedder):
        return embedding_model

    if "openai" in str(type(embedding_model)):
        from ._openai import OpenAIBackend
        return OpenAIBackend(embedding_model)
    # Sentence Transformer embeddings
    if "sentence_transformers" in str(type(embedding_model)):
        from ._sentencetransformers import SentenceTransformerBackend
        return SentenceTransformerBackend(embedding_model)

    # Create a Sentence Transformer model based on a string
    if isinstance(embedding_model, str):
        from ._sentencetransformers import SentenceTransformerBackend
        return SentenceTransformerBackend(embedding_model)

    
    from ._sentencetransformers import SentenceTransformerBackend
    return SentenceTransformerBackend("paraphrase-multilingual-MiniLM-L12-v2")

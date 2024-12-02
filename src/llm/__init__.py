from src._utils import NotInstalled
from src.llm._base import BaseLLM


# TextGeneration
try:
    from src.llm._textgeneration import TextGeneration
except ModuleNotFoundError:
    msg = "`pip install src` \n\n"
    TextGeneration = NotInstalled("TextGeneration", "src", custom_msg=msg)

# OpenAI Generator
try:
    from src.llm._openai import OpenAI
except ModuleNotFoundError:
    msg = "`pip install openai` \n\n"
    OpenAI = NotInstalled("OpenAI", "openai", custom_msg=msg)

# Cohere Generator
try:
    from src.llm._cohere import Cohere
except ModuleNotFoundError:
    msg = "`pip install cohere` \n\n"
    Cohere = NotInstalled("Cohere", "cohere", custom_msg=msg)

# LangChain Generator
try:
    from src.llm._langchain import LangChain
except ModuleNotFoundError:
    msg = "`pip install langchain` \n\n"
    LangChain = NotInstalled("langchain", "langchain", custom_msg=msg)

# LiteLLM
try:
    from src.llm._litellm import LiteLLM
except ModuleNotFoundError:
    msg = "`pip install litellm` \n\n"
    LiteLLM = NotInstalled("LiteLLM", "litellm", custom_msg=msg)


__all__ = [
    "BaseLLM",
    "Cohere",
    "OpenAI",
    "TextGeneration",
    "LangChain",
    "LiteLLM"
]

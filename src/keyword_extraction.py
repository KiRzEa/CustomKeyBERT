# Standard library imports
import os
from pathlib import Path
from typing import Any, Optional
from xml.dom import minidom
import xml.etree.ElementTree as ET

# Third-party library imports
from bs4 import BeautifulSoup
import pandas as pd
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# OpenAI-specific imports
from openai import AzureOpenAI

# Local application/library-specific imports
from src import KeyBERT, KeyLLM
from src.llm import OpenAI


from dotenv import load_dotenv
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR")
DATA_DIR  = os.getenv("DATA_DIR")

class Keyword(BaseModel):
    keyword: str
    score: Optional[float] = None
    embeddings: Optional[Any] = None
    file: str
    file_path: str
    component: Optional[str] = None

class Component(BaseModel):
    component_id: Any
    component: str
    keyword: str
    file: str
    score: float

class InputComponent(BaseModel):
    id: Any
    description: str
    embeddings: Any

class KeywordExtractor:
    def __init__(self, embed_model, llm=None):
        if isinstance(embed_model, str):
            self.embed_model = SentenceTransformer(embed_model)
        else:
            self.embed_model = embed_model
        if llm:
            if isinstance(llm, AzureOpenAI):
                self.llm = OpenAI(client=llm, model='gpt-4o', chat=True)
            elif isinstance(llm, OpenAI):
                self.llm = llm
            self.model = KeyLLM(llm=self.llm)
        else:
            self.model = KeyBERT(model=self.embed_model)
        
        self.files = self.get_html_files()

        self.doc_keywords = None
        # self.doc_keywords = self.get_doc_keywords()


    def get_html_files(self):
        # Path to the folder containing files
        folder_path = Path(DATA_DIR)  # Replace with your folder path

        # List to store HTML file paths
        self.files = []

        # Iterate through all files in the folder
        for file_name in os.listdir(folder_path):
            # Check if the file has a .html extension
            if file_name.endswith(".html"):
                # Add the full path of the .html file to the list
                self.files.append(folder_path / file_name)

        return self.files

    # Path to the HTML file
    def get_text(self, path):
        # Read the HTML file
        with open(path, "r", encoding="utf-8") as file:
            html_content = file.read()

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract all textual data
        textual_content = []

        # Function to concatenate content of a tag and its children
        def extract_text(tag):
            return ' '.join(tag.stripped_strings)

        # Extract paragraphs, list items, and divs for textual data
        for tag in soup.find_all(['p', 'li', 'span']):
            if tag.name == "li":
                # Concatenate text within <li> and its nested tags
                text = extract_text(tag)
            else:
                # For other tags, keep text as is
                text = tag.get_text(separator='\n', strip=True).strip("“")

            if text:
                textual_content.append(text)

        return ' '.join(textual_content)

    def embed(self, text):
        embedding =  self.embed_model.encode(text).reshape(1, -1)
        return embedding
    
    def get_components(self, path):
        self.components = pd.read_excel(path)
        return self.components

    def get_doc_keywords(self):
        self.total_keywords = []

        for file in tqdm(self.files, desc='[INFO] Extracting keywords...'):
            textual_content = self.get_text(file)
            if isinstance(self.model, KeyLLM):
                doc_keywords = self.model.extract_keywords(
                    textual_content,
                )
                filtered_keywords = [Keyword(keyword=keyword, score=1.0, file=file.name, file_path=str(file)) for keyword in doc_keywords[0]]
                # for keyword in filtered_keywords:
                #     keyword.embeddings = self.embed(keyword.keyword)
            elif isinstance(self.model, KeyBERT):
                doc_keywords = self.model.extract_keywords(
                    textual_content,
                    top_n=100,
                    keyphrase_ngram_range=(1, 3),
                )
                filtered_keywords = [Keyword(keyword=keyword, score=score, file=file.name, file_path=str(file)) for keyword, score in doc_keywords if score > 0.6]

                for keyword in filtered_keywords:
                    keyword.embeddings = self.embed(keyword.keyword)
            self.total_keywords.extend(filtered_keywords)


        return self.total_keywords

    def get_related_files(self, component: InputComponent, top_k: int = 10, threshold: float = 0.6):
        scores = []
        for keyword in self.doc_keywords:
            try:
                score = cosine_similarity(
                    component.embeddings,
                    keyword.embeddings
                )
            except:
                score = cosine_similarity(
                    component.embeddings,
                    self.embed(keyword.keyword)
                )

            scores.append(Component(
                component_id=component.id,
                component=component.description,
                keyword=keyword.keyword,
                file=keyword.file,
                score=score
            ))
        scores = sorted(scores, key=lambda x: x.score, reverse=True)
        results = []
        files = []
        for score in scores:
            if score.file not in files and score.score > threshold:
                results.append(score)
                files.append(score.file)
        return results[:top_k]
    
    def generate_component_xml(self, components, file_name):
        """
        Generates an XML file with component data.

        Args:
            components (list of dict): List of components with keys 'id', 'desc', 'confidence'.
            file_name (str): File name used to output .xml file.
        """
        # Root element
        root = ET.Element("Components")
        
        for component in components:
            # Create a Component element
            comp_elem = ET.SubElement(root, "Component")
            
            # Add child elements
            ET.SubElement(comp_elem, "Comp_ID").text = str(component['ID'])
            ET.SubElement(comp_elem, "Comp_Desc").text = component.description
            ET.SubElement(comp_elem, "Comp_Confidence").text = str(component.score)
        
        # Convert the tree to a string
        rough_string = ET.tostring(root, encoding="utf-8")

        # Parse the string and pretty print
        parsed_xml = minidom.parseString(rough_string)
        pretty_xml = parsed_xml.toprettyxml(indent="  ")

        # Write to an XML file
        output_file = file_name.replace('.html', '.xml')
        with open(f"result/{output_file}", "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        print(f"Pretty-printed XML file generated: {output_file}")
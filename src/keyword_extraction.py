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
from sentencex import segment
from tqdm import tqdm
from src.prompts import EVALUATION_PROMPT

# OpenAI-specific imports
from openai import AzureOpenAI

# Local application/library-specific imports
from src import KeyBERT, KeyLLM
from src.llm import OpenAI


from dotenv import load_dotenv
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR")
DATA_DIR  = os.getenv("DATA_DIR")

gpt_client = AzureOpenAI(
    api_key='8f257f6f430842f29012b3bccf8cf5f9',
    azure_endpoint='https://bosch-cr-openai.openai.azure.com/',
    azure_deployment='gpt-4o',
    api_version='2024-02-15-preview'
)

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

def get_components_by_file(results, file):
    components = list(results[file.name]['components'].keys())
    keywords = list(results[file.name]['keywords'].keys())
    return components, keywords

sorted_files = ['K00.5928.11_X-4HYXBX2540B_en.html',
'K00.5928.11_X-K6D34843_en.html',
'K00.5928.16_X-1X0V7W55OBG_en.html',
'K00.5928.16_x-3id07ma_en.html',
'K00.5928.19_X-HST6GXOU3X_en.html',
'K00.5928.19_X-R5WY9VJC2R_en.html',
'K00592810003_en.html',
'K00592810005_en.html',
'K00592810007_en.html',
'K00592810013_en.html',
'K00592810014_en.html',
'K00592810018_en.html',
'K00592810022_en.html',
'K00592810804_en.html',
'K00592810805_en.html',
'K00592810806_en.html',
'K00592810831_en.html',
'K00592810832_en.html',
'K00592810833_en.html']

class KeywordExtractor:
    def __init__(self, embed_model=None, judge_model=None, llm=None):
        self.judge_model = judge_model
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
            if self.embed_model:
                self.model = KeyBERT(model=self.embed_model)
            else:
                self.model = None
        
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
        embedding =  self.judge_model.embed(text)
        return embedding
    
    def get_components(self, path):
        self.components = pd.read_excel(path)
        return self.components

    def get_doc_keywords(self):
        self.doc_keywords = []

        for file in tqdm(self.files, desc='[INFO] Extracting keywords...'):
            textual_content = self.get_text(file)
            if isinstance(self.model, KeyLLM):
                docs = self.model.extract_keywords(
                    textual_content,
                )
                filtered_keywords = [Keyword(keyword=keyword, score=1.0, file=file.name, file_path=str(file)) for doc_keywords in docs for keyword in doc_keywords]
                # for keyword in filtered_keywords:
                #     keyword.embeddings = self.embed(keyword.keyword)
            elif isinstance(self.model, KeyBERT):
                doc_keywords = self.model.extract_keywords(
                    list(segment("en", textual_content)),
                    top_n=100,
                    keyphrase_ngram_range=(1, 3),
                )
                filtered_keywords = [Keyword(keyword=keyword, score=score, file=file.name, file_path=str(file)) for keyword, score in doc_keywords if score > 0.6]

            for keyword in filtered_keywords:
                keyword.embeddings = self.embed(keyword.keyword)
            self.doc_keywords.extend(filtered_keywords)


        return self.doc_keywords

    def get_related_files(self, component: InputComponent, top_k: int = 10, threshold: float = 0.6):
        scores = []
        for keyword in self.doc_keywords:
            try:
                score = cosine_similarity(
                    component.embeddings.reshape(1, -1),
                    keyword.embeddings.reshape(1, -1)
                )
            except:
                score = cosine_similarity(
                    self.embed(component.description).reshape(1, -1),
                    self.embed(keyword.keyword).reshape(1, -1)
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
        # files = []
        for score in scores:
            # if score.file not in files and score.score > threshold:
            if score.score > threshold:
                results.append(score)
                # files.append(score.file)
        return results[:top_k]

    def predict(self, components, threshold=0.5):
        final_results = []
        for idx, comp in tqdm(components.iterrows()):
            input_comp = InputComponent(
                id=comp['ID'],
                description=comp['Description (ENG)'],
                embeddings=comp['embeddings']
            )
            result = self.get_related_files(input_comp, threshold=threshold)
            final_results.extend(result)

        file_results = {}
        for file in tqdm(sorted_files):
            file_keywords = {}
            file_components = {}
            for result in final_results:
                if result.file == file:
                    if not file_keywords.get(result.keyword, None):
                        file_keywords[result.keyword] = [result.model_dump()]
                    else:
                        file_keywords[result.keyword].append(result.model_dump())
                    if not file_components.get(result.component, None):
                        file_components[result.component] = [result.model_dump()]
                    else:
                        file_components[result.component].append(result.model_dump())

            file_results[file] = {'keywords': file_keywords, 'components': file_components}   
        return file_results
    def evaluate(self, results):
        responses = {file.name: [] for file in self.files}

        for file in tqdm(self.files):
            document=self.get_text(file)
            for extracted_component in get_components_by_file(results, file)[0]:
                prompt_call = EVALUATION_PROMPT.format(document=document, component=extracted_component,
                                                    filename=str(file))
                messages = [{
                    'role': 'user', 'content': prompt_call
                }]
                response = gpt_client.chat.completions.create(
                    model='gpt-4o',
                    messages=messages,
                    temperature=0.0
                )
                responses[file.name].append(response.choices[0].message.content)
        
        return responses
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
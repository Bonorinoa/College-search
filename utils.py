# general imports
import os
import pandas as pd
from datetime import datetime
import streamlit as st

# asynchronous scraping
import asyncio
import html2text

# google scholar
from serpapi import GoogleSearch

# langchain
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_openai import ChatOpenAI

## Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["SERP_API_KEY"] = st.secrets["SERP_API_KEY"]


#TODO: Populate placeholder functions with actual code 


def load_llm(provider="openai", max_tokens=100, temperature=0.5):
    """
    Load the language model from the specified provider.
    Openai loads gpt-3.5-turbo by default.
    Huggingface can be added later.

    input:
        provider: str, "openai" or "huggingface"
        max_tokens: int, maximum number of tokens to generate
        temperature: float, temperature for sampling
    output:
        llm: Langchain language model object
    """

    if provider == "openai":
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", max_tokens=max_tokens, temperature=temperature
        )

    # elif provider == "huggingface":
    # Load the model from Hugging Face

    return llm

# Function to fetch google scholar papers
## should return a list of JSON objects that have the key "authors"



# Function to fetch the authors' google scholar profiles
## should return a list of JSON objects that we can parse to get the authors' institutions and research interests



# Function to parse authors' institutions and research interests 
## should return a list of strings that can be displayed in the dropdown menu



# Function to fetch university admissions data from the database
## should take in the institution chosen by the user and return the university admissions data
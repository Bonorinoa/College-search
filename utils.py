# general imports
import os
import pandas as pd
from datetime import datetime
import streamlit as st
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


# TODO: Populate placeholder functions with actual code


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


# Class to search google scholar
## query function for searching papers
## user function for searching authors
## returns JSON objects
class SearchScholar:
    def __init__(self):
        self.api_key = os.getenv("SERP_API_KEY")
        if not self.api_key:
            raise ValueError("SERP_API_KEY is not set in environment variables.")

    def query(self, query, as_ylo=2018, as_yhi=2024, start=0, num=10):
        params = {
            "engine": "google_scholar",
            "api_key": self.api_key,
            "q": query,
            "as_ylo": as_ylo,
            "as_yhi": as_yhi,
            "hl": "en",
            "start": start,
            "num": num,  # limited to 20
            "output": "json",
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])

        return results, organic_results

    def user(self, authors_list):
        for author in authors_list:
            params = {
                "engine": "google_scholar_author",
                "api_key": self.api_key,
                "author_id": author["author_id"],
                "output": "json",
            }

            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                author_data = results.get("author", {})
                author["affiliations"] = author_data.get(
                    "affiliations", "No Affiliation Found"
                )
                # Extracting interests (could be multiple)
                interests = author_data.get("interests", [])
                author["interests"] = [interest["title"] for interest in interests]
            except Exception as e:
                print(f"Failed to fetch information for author {author['name']}: {e}")
                author["affiliations"] = "No profile found"

        return authors_list


# Class to parse JSON combining the functions below
## parse entire JSON to authors json
## parse authors json to list
class JSONParser:
    def __init__(self):
        pass

    # Function to get author information from the JSON data
    ## will still return json, but just of the authors
    def extract_author_json(self, json_data):
        authors_info = []

        # Iterate through each paper
        for entry in json_data:
            publication_info = entry.get("publication_info", {})
            authors = publication_info.get("authors", [])

            # Iterate through each author in the paper
            for author in authors:
                # Desired information to extract
                author_details = {
                    "name": author.get("name", "No Name Provided"),
                    "link": author.get("link", "No Link Provided"),
                    "serpapi_scholar_link": author.get(
                        "serpapi_scholar_link", "No SerpApi Link Provided"
                    ),
                    "author_id": author.get("author_id", "No Author ID Provided"),
                }
                # Add author info
                authors_info.append(author_details)

        return authors_info

    # Function to parse authors' informatin from the JSON data
    ## should return a list of strings that can be displayed in the dropdown menu
    @staticmethod
    @st.cache_data
    def author_json_to_list(authors_list):
        formatted_list = []

        # Iterate through authors for name, affliation, and research interests
        for author in authors_list:
            formatted_str = f"{author['name']} --- {author['affiliations']}"
            formatted_list.append(formatted_str)

        return formatted_list


# Function to fetch university admissions data from the database
## should take in the institution chosen by the user and return the university admissions data
@st.cache_data
def load_admissions_data():
    # Load the admissions data from the database
    admissions_data = pd.read_csv("data/Merged_Admissions_Data.csv")

    return admissions_data

def extract_state_universities(institution):
    llm = load_llm(max_tokens=15, temperature=0.5)

    prompt = f"""
            The following string contains the name of an accredited university. 
            Extract the name of the university from the string, if it is abbreviated please expand it to match the legal name: 
            
            {institution}
            
            You must return the two-letter abbreviation of the state where the university is located (i.e., Washington = WA, or Virginia = VA).
            
            STABBR: 
        """

    state = llm.invoke(prompt)

    return state.content

def fetch_admissions_state_data(institution):
    # Load the admissions data
    admissions_data = load_admissions_data()
    
    state = extract_state_universities(institution)
    
    st.write("The extracted state is:", state)

    # Filter the data based on the selected institution
    try:
        institution_data = admissions_data[admissions_data["STABBR"] == state]
        if institution_data.empty:
            st.warning("No data found for the selected institution.")
            university = st.text_input("Enter the name of the institution to fetch the data:")
            institution_data = admissions_data[admissions_data["INSTNM"] == university]
        
    except Exception as e:
        st.warning(f"We couldn't fetch your school: {e}")
        

    return institution_data
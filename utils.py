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
        pass

    def query(self, query, as_ylo=2018, as_yhi=2024, start=0, num=10):
        params = {
            "engine": "google_scholar",
            "api_key": os.getenv("SERP_API_KEY"),
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

    def author_profiles(self, authors_info):

        # author_profiles = [
        #     {"name": "", "affiliations": "", "website": "", "interests": []}
        # ]
        author_profiles = []

        for author in authors_info:
            params = {
                "engine": "google_scholar_author",
                "api_key": os.getenv("SERP_API_KEY"),
                "author_id": author["author_id"],
                "output": "json",
            }

            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                author_data = results.get("author", {})

                author_profiles.append(
                    {
                        "name": author.get(
                            "name", "No Name Found"
                        ),  # "name": "No Name Found
                        "website": author_data.get("website", "No Website Found"),
                        "affiliations": author_data.get(
                            "affiliations", "No Affiliation Found"
                        ),
                        "interests": author_data.get("interests", []),
                    }
                )

            except Exception as e:
                st.warning(f"Couldn't fetch author's profile: {e}")

        return author_profiles


# Class to parse JSON combining the functions below
## parse entire JSON to authors json
## parse authors json to list
class JSONParser:
    def __init__(self):
        pass

    # Function to get author information from the JSON data
    ## will still return json, but just of the authors
    def extract_authors(self, json_data):
        # authors_info = [{"name": "", "author_id": ""}]
        authors_info = []

        # Iterate through each paper
        for entry in json_data:

            authors = entry.get("publication_info", "No publications").get(
                "authors", "no authors"
            )

            if authors != "no authors":
                for author in authors[:2]:
                    authors_info.append(
                        {
                            "name": author.get("name", "No Name Found"),
                            "author_id": author.get("author_id", "No ID Found"),
                        }
                    )

        return authors_info

    # Function to parse authors' information from the JSON data
    ## should add a string description for each author that can be displayed in the dropdown menu
    # @staticmethod
    # @st.cache_data
    def author_string(self, author_profiles):
        # Iterate through authors to add a 'parsed_string' trait
        for author in author_profiles:
            formatted_str = f"{author['name']} --- {author.get('affiliations', 'No Affiliation Found')}"
            author["parsed_string"] = formatted_str

        return authors_profiles


def build_author_description(author_profile):
    llm = load_llm(max_tokens=250, temperature=0.75)

    prompt = f"""
            The following information contains the name of an author, interests, and their affiliations.\n\n
            
            Name: {author_profile['name']}\n
            Website url: {author_profile['website']}\n
            Affiliations: {author_profile['affiliations']}\n
            Interests: {author_profile['interests']}\n\n
            
            Please provide a detailed and factual (based on provided data) description of the author's research interests and affiliations. 
            The target audience are undergraduate and graduate students looking to learn more about the author's research.
            Include any links to the author's website.
            The description should be informative.
            """

    description = llm.invoke(prompt)

    return description.content


def build_university_description(university_data):
    llm = load_llm(max_tokens=450, temperature=0.85)

    prompt = f"""
            The following information contains the name of a university, its admissions data, and other relevant information.\n\n
            
            Name: {university_data['INSTNM']}\n
            Website: {university_data['WEBADDR']}\n
            Admissions website: {university_data['ADMINURL']}\n
            Funding website: {university_data['FAIDURL']}\n
            Disability services website: {university_data['DISAURL']}\n
            Highest degree offered: {university_data['HLOFFER']}\n
            SAT Math 75th percentile: {university_data['SATMT75']}\n
            SAT Reading 75th percentile: {university_data['SATVR75']}\n
            ACT 75th percentile: {university_data['ACTCM75']}\n
            Applicants in 2022: {university_data['APPLCN']}\n
            Admitted in 2022: {university_data['ADMSSN']}\n\n
            
            Please provide a detailed and factual (based on provided data) description of the university.
            The target audience are undergraduate and graduate students looking to learn about admissions criteria and relevant links.
            The description should be informative and comprehensive.
            """

    description = llm.invoke(prompt)

    return description.content


def search_institution_description(institution):
    llm = load_llm(max_tokens=450, temperature=0.85)

    prompt = f"""
            The following information contains the name of an institution.\n\n
            
            Institution: {institution}\n
            
            Please provide a detailed and factual description of the university.
            The target audience are undergraduate and graduate students looking to learn about admissions criteria and relevant links.
            The description should be informative and comprehensive.
            """

    description = llm.invoke(prompt)

    return description.content


def build_cold_email(author_profile):
    llm = load_llm(max_tokens=450, temperature=0.85)

    prompt = f"""
            Please write the structure for a cold email to the following researcher\n\n
            
            Name: {author_profile['name']}\n
            Affiliations: {author_profile['affiliations']}\n
            Interests: {author_profile['interests']}\n\n
            
            The email should be professional and concise, yet enthusiastic.
            Show that you have done your research on the author's work and that you are interested in learning more about the research.
            Do not assume the gender of the author.
            
            ------
            
            Once you have written the email, please provide a list of three questions you would ask the author in a follow-up email.
            Finally provide a list of three helpful reminders (e.g., to thank the author for their time, to read their work before writing the email, and to look for their email on the website). )
        """

    email = llm.invoke(prompt)

    return email.content


# Function to fetch university admissions data from the database
## should take in the institution chosen by the user and return the university admissions data
# @st.cache_data
def load_admissions_data():
    # Load the admissions data from the database
    admissions_data = pd.read_csv("data/Merged_Admissions_Data.csv")

    return admissions_data


def extract_state_universities(institution):
    llm = load_llm(max_tokens=2, temperature=0.5)

    prompt = f"""
            The following string contains the name of an accredited university.
            
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
    print("The extracted state is:", repr(state))

    # Determine which list to use based on the existence of the state
    if state in admissions_data["STABBR"].values:
        institutions_list = admissions_data[admissions_data["STABBR"] == state][
            "INSTNM"
        ].tolist()

        # Find top 3 similarity
        top3_universities = find_university(institutions_list, institution)

        # Build dataframe for top 3 universities
        universities_data = pd.DataFrame()
        for university in top3_universities:
            university_data = get_university_data(university)
            universities_data = pd.concat([universities_data, university_data])
    else:
        st.write(
            """WARNING: We currently don't support international universities and are unable to fetch statistics at this time.
            However, we will perform an information search on the institution you provided.
            """
        )
        universities_data = pd.DataFrame()

    return universities_data, institution


def find_university(state_data, author_institution):
    """
    Takes in the list of universities in a state and returns the university with the highest cosine similarity with the author's institution.
    inputs:
        state_data: list of universities in a state
        author_institution: institution of the author
    outputs:
        university: the university with the highest cosine similarity with the author's institution
    """

    # Creating the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Including the author's institution in the list for vectorization
    all_text = state_data + [author_institution]

    # Transforming the text to TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(all_text)

    # Calculating cosine similarity between the author's institution and all universities in the state
    # The last entry in tfidf_matrix is the author's institution
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # get the indices for the top three cosine similarities
    top_indices = cosine_similarities.argsort()[0][::-1][:3]

    # filter state data to get the top three universities
    top_universities = [state_data[i] for i in top_indices]

    # Returning the university with the highest cosine similarity
    return top_universities


def get_university_data(university):
    # Load the admissions data
    admissions_data = load_admissions_data()

    # Filter the data based on the selected institution
    try:
        university_data = admissions_data[admissions_data["INSTNM"] == university]

    except Exception as e:
        st.warning(f"We couldn't fetch your school: {e}")

    return university_data

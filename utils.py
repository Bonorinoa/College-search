import os
import pandas as pd
import asyncio
from serpapi import GoogleSearch
from datetime import datetime
import html2text
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_openai import ChatOpenAI

from langchain_openai import ChatOpenAI


# Search Google Scholar
def search_scholar(params):

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]

    return results, organic_results


# Define a function to parse the JSON data and return a DataFrame
def parse_json_to_df(json_data):
    # List to hold each row of the dataframe
    rows = []
    # Current timestamp to add to each row
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Iterate through each entry in the JSON data
    for entry in json_data:
        # Extract required information
        title = entry.get("title", "")
        link = entry.get("link", "")
        citations = entry.get("inline_links", {}).get("cited_by", {}).get("total", 0)
        versions = entry.get("inline_links", {}).get("versions", {}).get("total", 0)
        cluster_link = entry.get("inline_links", {}).get("versions", {}).get("link", "")

        # there are still more interesting objects to extract from organic_results

        # Append the extracted information as a new row
        rows.append([title, now, link, citations, versions, cluster_link])

    # Create a DataFrame
    df = pd.DataFrame(
        rows,
        columns=[
            "Title",
            "Time of Extraction",
            "Link",
            "Citations",
            "Versions",
            "Cluster Link",
        ],
    )

    return df


async def do_webscraping(link):
    """
    Function to search a link asynchronously. Returns a JSON load with extracted HTML objects parsed from each link.
    Parameters
        link: A single url to search. We pass several link in a loop so that we can do it async
    Returns:
        doc: A JSON (python dictionary) object with relevant information parsed from the HTML scraped from the link.
    """
    try:
        urls = [link]
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()

        html2text_transformer = Html2TextTransformer()
        docs_transformed = html2text_transformer.transform_documents(docs)

        if docs_transformed != None and len(docs_transformed) > 0:
            metadata = docs_transformed[0].metadata
            title = metadata.get("title", "")
            content = docs_transformed[0].page_content

            doc = {
                "title": title,
                "metadata": metadata,
                "page_content": html2text.html2text(content),
            }
            return doc
        else:
            return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# Async webscraping
async def collect_webscraping(df):
    tasks = [do_webscraping(link) for link in df["Link"]]
    tasks_output = await asyncio.gather(*tasks)
    # Filter out None responses if necessary
    structured_responses = [
        response for response in tasks_output if response is not None
    ]
    return structured_responses


"""LLM Utils"""


def format_paper_content(paper_content: str) -> str:
    """
    Formats the paper content based on the specified condition.

    Args:
        paper_content (str): The content of the paper.

    Returns:
        str: Formatted paper content.
    """
    word_list = paper_content.split()
    if len(word_list) > 10000:
        # Concatenate the top and bottom 3000 words
        ## Rylie: I changed this from 5000 to 3000 because I got a token length error
        ## I wonder if there's a way to shorten based on tokens instead of words
        formatted_content = (
            " ".join(word_list[:3000]) + " ... " + " ".join(word_list[-3000:])
        )
    else:
        formatted_content = paper_content

    return formatted_content


def get_info(docs, paper_test_idx: int):

    paper_test_link = docs[paper_test_idx]["metadata"]["source"]
    # Rylie: Should this be feeding the page_content, or instead ['metadata']['title']?
    paper_test_content = docs[paper_test_idx]["page_content"]

    print(len(paper_test_content.split(" ")))

    ## method highly dependent on capturing the right block of text that contains the authors
    ## vector retrieval + reranking might help here
    prompt = f"""
        In the following paper, to the best of your ability, identify the main authors and their 
        corresponding schools. Only give the school name, no additional information. Give your 
        response In an array of tuples as if inputting to code. Here's an example of how I'd like you to format your response:
        [(Rylie Weaver, Claremont Graduate University), (Augusto Gonzalez-Bonorino, AllenAI Lab), and so on...].

        \n\n
        Paper content
        \n
        {format_paper_content(paper_test_content)}

        --------------
        Authors/Institutions:
    """

    gpt = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # GPT4 has a 4k context window, effort better spent in trying to retrieve relevant documents
        temperature=0.5,
        max_tokens=150,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    response = gpt.invoke(prompt)

    return response.content, paper_test_link


# Define a function to parse the JSON data and return a DataFrame
def parse_json_to_df(json_data):
    # List to hold each row of the dataframe
    rows = []
    # Current timestamp to add to each row
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Iterate through each entry in the JSON data
    for entry in json_data:
        # Extract required information
        title = entry.get("title", "")
        link = entry.get("link", "")
        citations = entry.get("inline_links", {}).get("cited_by", {}).get("total", 0)
        versions = entry.get("inline_links", {}).get("versions", {}).get("total", 0)
        cluster_link = entry.get("inline_links", {}).get("versions", {}).get("link", "")

        # there are still more interesting objects to extract from organic_results

        # Append the extracted information as a new row
        rows.append([title, now, link, citations, versions, cluster_link])

    # Create a DataFrame
    df = pd.DataFrame(
        rows,
        columns=[
            "Title",
            "Time of Extraction",
            "Link",
            "Citations",
            "Versions",
            "Cluster Link",
        ],
    )

    return df


async def do_webscraping(link):
    """
    Function to search a link asynchronously. Returns a JSON load with extracted HTML objects parsed from each link.
    Parameters
        link: A single url to search. We pass several link in a loop so that we can do it async
    Returns:
        doc: A JSON (python dictionary) object with relevant information parsed from the HTML scraped from the link.
    """
    try:
        urls = [link]
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()

        html2text_transformer = Html2TextTransformer()
        docs_transformed = html2text_transformer.transform_documents(docs)

        if docs_transformed != None and len(docs_transformed) > 0:
            metadata = docs_transformed[0].metadata
            title = metadata.get("title", "")
            content = docs_transformed[0].page_content

            doc = {
                "title": title,
                "metadata": metadata,
                "page_content": html2text.html2text(content),
            }
            return doc
        else:
            return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# Async webscraping
async def collect_webscraping(df):
    tasks = [do_webscraping(link) for link in df["Link"]]
    tasks_output = await asyncio.gather(*tasks)
    # Filter out None responses if necessary
    structured_responses = [
        response for response in tasks_output if response is not None
    ]
    return structured_responses


"""LLM Utils"""


def format_paper_content(paper_content: str) -> str:
    """
    Formats the paper content based on the specified condition.

    Args:
        paper_content (str): The content of the paper.

    Returns:
        str: Formatted paper content.
    """
    word_list = paper_content.split()
    if len(word_list) > 10000:
        # Concatenate the top and bottom 3000 words
        ## Rylie: I changed this from 5000 to 3000 because I got a token length error
        ## I wonder if there's a way to shorten based on tokens instead of words
        formatted_content = (
            " ".join(word_list[:3000]) + " ... " + " ".join(word_list[-3000:])
        )
    else:
        formatted_content = paper_content

    return formatted_content


def get_info(docs, paper_test_idx: int):

    paper_test_link = docs[paper_test_idx]["metadata"]["source"]
    # Rylie: Should this be feeding the page_content, or instead ['metadata']['title']?
    paper_test_content = docs[paper_test_idx]["page_content"]

    print(len(paper_test_content.split(" ")))

    ## method highly dependent on capturing the right block of text that contains the authors
    ## vector retrieval + reranking might help here
    prompt = f"""
        In the following paper, to the best of your ability, identify the main authors and their 
        corresponding schools. Only give the school name, no additional information. Give your 
        response In an array of tuples as if inputting to code. Here's an example of how I'd like you to format your response:
        [(Rylie Weaver, Claremont Graduate University), (Augusto Gonzalez-Bonorino, AllenAI Lab), and so on...].

        \n\n
        Paper content
        \n
        {format_paper_content(paper_test_content)}

        --------------
        Authors/Institutions:
    """

    gpt = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # GPT4 has a 4k context window, effort better spent in trying to retrieve relevant documents
        temperature=0.5,
        max_tokens=150,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    response = gpt.invoke(prompt)

    return response.content, paper_test_link


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


def load_llm(provider="openai",
             max_tokens=100,
             temperature=0.5):
    
    '''
    Load the language model from the specified provider.
    Openai loads gpt-3.5-turbo by default.
    Huggingface can be added later.
    
    input:
        provider: str, "openai" or "huggingface"
        max_tokens: int, maximum number of tokens to generate
        temperature: float, temperature for sampling
    output:
        llm: Langchain language model object
    '''

    if provider == "openai":
        llm = ChatOpenAI(model="gpt-3.5-turbo", 
                            max_tokens=max_tokens,
                            temperature=temperature)
        
    #elif provider == "huggingface":
        # Load the model from Hugging Face
        
    return llm
from serpapi import GoogleSearch

from langchain_openai import ChatOpenAI

# Search Google Scholar
def search_scholar(params):

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]

    return results, organic_results


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
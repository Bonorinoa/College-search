from serpapi import GoogleSearch


# Search Google Scholar
def search_scholar(params):

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]

    return results, organic_results

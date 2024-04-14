import streamlit as st
from time import sleep
import re

from utils import *


# streamlit fragment functions
@st.experimental_fragment
def fragment_function(authors_info):
    # dropdown is populated with the parsed data
    st.markdown("## Select the author-institution pair")

    # get 'parsed_string' from the authors_info for the dropdown
    parsed_data = [author["parsed_string"] for author in authors_info]

    user_choice = st.selectbox("Author-Institution Pair", parsed_data)

    st.write(f"User choice: {user_choice}")

    # button to submit the choice
    if st.button("Make Selection"):
        with st.spinner("Fetching data..."):
            # st.balloons()

            # map the user choice to the author
            ## index 0 in case there are multiple
            chosen_author = [
                author
                for author in authors_info
                if author["parsed_string"] == user_choice
            ][0]

            # Call chat to get description of the author and their institution
            paragraph_llm = load_llm(max_tokens=200, temperature=0.5)
            prompt = f"""
                    I'm interesting in learning about researchers and research institutions. 
                    Tell me about {chosen_author["name"]} and their research interests, which 
                    I believe include: {chosen_author["interests"]}. Also tell me about their 
                    insitution: {chosen_author["affiliations"]}.
                    """
            paragraph = paragraph_llm.invoke(prompt)
            st.write(paragraph.content)

            # get the author's affiliation
            author_affiliation = chosen_author.get(
                "affiliations", "No Affiliation Found"
            )

            universities_data = fetch_admissions_state_data(author_affiliation)

            st.dataframe(universities_data)


@st.experimental_fragment
def fetch_institution_data():

    university = st.text_input("Enter the name of the institution to fetch the data:")

    if st.button("Get univesity admissions data"):
        with st.spinner("Fetching data..."):

            uni_data = get_university_data(university)
            st.dataframe(uni_data)


def main():

    # streamlit state variables

    st.title("Hello - 5C Hackathon")
    st.write(
        "This is a simple web app to demonstrate the deployment of a machine learning model using Streamlit."
    )

    # user input
    st.sidebar.header("User Input Parameters")
    st.sidebar.write("What do you want to study/research?")

    # get user's input (research interests)
    user_input = st.sidebar.text_area("Describe your research interests:", "...")

    # Instantiate the classes in the beginning
    search_scholar = SearchScholar()
    json_parser = JSONParser()

    if len(user_input) > 5:

        # user user input to fetch google scholar papers
        with st.spinner("Fetching papers..."):
            # Perform a query search
            results, organic_results = search_scholar.query(user_input)

        # get author's info from papers
        with st.spinner("Fetching authors..."):
            authors_info = json_parser.extract_authors(organic_results)
            st.success("Authors fetched successfully!")

        # add info from google scholar profiles
        with st.spinner("Fetching authors' profiles..."):
            authors_info = search_scholar.author_profiles(authors_info)
            st.success("Authors' profiles fetched successfully!")

        # get authors' institutions and research interests
        with st.spinner("Parsing authors' profiles..."):
            authors_info = json_parser.author_string(
                authors_info
            )  # adds pairs of author-insitution
            st.success("Authors' profiles parsed successfully!")

        ## CACHE THE RESULTS CREATED UP TO THIS POINT

        # create list of strings to display in dropdown
        ## could be pairs of author-institution

        st.markdown("---")

        fragment_function(authors_info)

        # fetch_institution_data()


if __name__ == "__main__":
    main()

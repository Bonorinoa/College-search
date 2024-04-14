import streamlit as st
from time import sleep

from utils import *


# streamlit fragment functions
@st.experimental_fragment
def fragment_function(parsed_data):
    # dropdown is populated with the parsed data
    st.markdown("## Select the author-institution pair")

    user_choice = st.selectbox("Author-Institution Pair", parsed_data)

    st.write(f"User choice: {user_choice}")

    # button to submit the choice
    if st.button("Get univesities in state"):
        with st.spinner("Fetching data..."):
            # st.balloons()

            # user_choice is used here to fetch the data from the database or API
            ## We need information about the author from some public API
            ## Institution information can be fetched from the database

            user_choice = user_choice.split(" --- ")
            author_name = user_choice[0]
            author_affiliation = user_choice[1]

            universities_data = fetch_admissions_state_data(author_affiliation)
           
            st.dataframe(universities_data)
            
            # st.write("Thank you for using the web app!")

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

        # get papers with key "authors"
        with st.spinner("Fetching authors..."):
            authors_json = json_parser.extract_author_json(organic_results)
            st.success("Authors fetched successfully!")

        # find authors' google scholar profiles
        with st.spinner("Fetching authors' profiles..."):
            authors_info = search_scholar.user(authors_json)
            st.success("Authors' profiles fetched successfully!")

        # get authors' institutions and research interests
        with st.spinner("Parsing authors' profiles..."):
            authors_list = json_parser.author_json_to_list(
                authors_info
            )  # pairs of author-insitution
            st.success("Authors' profiles parsed successfully!")

        ## CACHE THE RESULTS CREATED UP TO THIS POINT

        # create list of strings to display in dropdown
        ## could be pairs of author-institution

        st.markdown("---")

        fragment_function(authors_list)
        
        #fetch_institution_data()


if __name__ == "__main__":
    main()

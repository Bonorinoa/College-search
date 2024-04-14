import streamlit as st
from time import sleep

from utils import *


if "user_choice" not in st.session_state:
    st.session_state.user_choice = None
    
if "uni_data" not in st.session_state:
    st.session_state.uni_data = None
    
if "author_profile" not in st.session_state:
    st.session_state.author_profile = None

# streamlit fragment functions
@st.experimental_fragment
def fragment_function(author_profiles):
    # dropdown is populated with the parsed data
    st.markdown("## Select the author-institution pair")
    
    pairs = []
    for author in author_profiles:
        formatted_str = f"{author['name']} --- {author.get('affiliations', 'No Affiliation Found')}"
        pairs.append(formatted_str)

    user_choice = st.selectbox("Author-Institution Pair", pairs)
    st.session_state.user_choice = user_choice

    st.write(f"User choice: {user_choice}")

    # button to submit the choice
    if st.button("Make Selection"):
        with st.spinner("Fetching data..."):
            # st.balloons()

            # map the user choice to the author
            author_name = user_choice.split(" --- ")[0]
            
            # get profile for the author
            for author in author_profiles:
                if author['name'] == author_name:
                    st.session_state.author_profile = author
            
            if st.session_state.author_profile is not None:
                description = build_author_description(st.session_state.author_profile)
            
                st.write(description)
                
def build_university_profile():
    
    if st.button("Show University Data"):
        
        with st.spinner("Fetching data..."):
            author_affiliation = st.session_state.user_choice.split(" --- ")[1]
    
            # dataframe
            universities_data = fetch_admissions_state_data(author_affiliation)
            
            st.write("Top 3 matches:")
            st.dataframe(universities_data)
            
            university = find_university(universities_data["INSTNM"].tolist(), author_affiliation)[0]
            
            st.write(f"Selected University: {university}")
            
            # get university data for selected affiliation
            uni_data = universities_data.loc[universities_data['INSTNM'] == university]
            st.session_state.uni_data = uni_data
                        
            uni_description = build_university_description(st.session_state.uni_data)
            
            st.write(uni_description)



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
    
    if st.session_state.author_profile is not None:
        email_sample = build_cold_email(st.session_state.author_profile)
        print(email_sample)
        st.sidebar.download_button("Download Email Sample", email_sample, 
                           file_name=f"{st.session_state['user_choice']}_SampleEmail.txt", key="email_sample")
            
            

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
            author_profiles = search_scholar.author_profiles(authors_info)
            st.success("Authors' profiles fetched successfully!")

        # get authors' institutions and research interests
        with st.spinner("Parsing authors' profiles..."):
            formatted_string = json_parser.author_string(
                author_profiles
            )  # adds pairs of author-insitution
            st.success("Authors' profiles parsed successfully!")

        ## CACHE THE RESULTS CREATED UP TO THIS POINT

        # create list of strings to display in dropdown
        ## could be pairs of author-institution

        st.markdown("---")

        fragment_function(author_profiles)
        
        if st.session_state.user_choice is not None:
            build_university_profile()

        # fetch_institution_data()


if __name__ == "__main__":
    main()

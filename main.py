import streamlit as st
from time import sleep

from utils import load_llm



def main():
    #### WEB APP Code ####
    st.title("Hello - 5C Hachathon")
    st.write(
        "This is a simple web app to demonstrate the deployment of a machine learning model using Streamlit."
    )


    # user input
    st.sidebar.header("User Input Parameters")
    st.sidebar.write("What do you want to study/research?")
    
    # get user's input (research interests)
    user_input = st.sidebar.text_area("Describe your research interests:", "...")

    if len(user_input) > 5:

        # user user input to fetch google scholar papers
        with st.spinner("Fetching papers..."):
            sleep(2)
            
            st.success("Results fetched successfully!")
        
        # get papers with key "authors"
        with st.spinner("Fetching authors..."):
            sleep(2)
            
            st.success("Authors fetched successfully!")
        
        
        # find authors' google scholar profiles
        with st.spinner("Fetching authors' profiles..."):
            sleep(2)
            
            st.success("Authors' profiles fetched successfully!")
        
        
        # get authors' institutions and research interests
        with st.spinner("Parsing authors' profiles..."):
            sleep(2)
            
            st.success("Authors' profiles parsed successfully!")
        
        
        # create list of strings to display in dropdown
        ## could be pairs of author-institution 
        parsed_data = [
            "Author 1 - Institution 1",
            "Author 2 - Institution 2",
            "Author 3 - Institution 3",
        ]

        st.markdown("---")

        # dropdown is populated with the parsed data
        st.markdown("## Select the author-institution pair")
        user_choice = st.selectbox("Author-Institution Pair", parsed_data)

        st.write(f"User choice: {user_choice}")

        # button to submit the choice
        if st.button("Submit"):
            st.write("You have submitted the choice")
            st.balloons()

            # user_choice is used here to fetch the data from the database or API
            ## We need information about the author from some public API
            ## Institution information can be fetched from the database

            st.write("Thank you for using the web app!")


if __name__ == "__main__":
    main()

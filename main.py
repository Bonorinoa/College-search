import streamlit as st
import os

from langchain_openai import ChatOpenAI

openai_key = st.secrets['OPENAI_API_KEY']
os.environ["OPENAI_API_KEY"] = openai_key

def main():
    #### WEB APP Code ####
    st.title("Hello - 5C Hachathon")
    st.write(
        "This is a simple web app to demonstrate the deployment of a machine learning model using Streamlit."
    )

    # user input
    st.sidebar.header("User Input Parameters")
    st.sidebar.write("What do you want to study/research?")
    user_input = st.sidebar.text_input("Describe your interest ", "Type here...")

    # Call Google Scholar API to get results, then parse with chat

    # Instead of printing it, this goes as input to LLM
    llm_test = ChatOpenAI(model="gpt-3.5-turbo", 
                          max_tokens=100)
    llm_response = llm_test.invoke(f"Create a google scholar search query to fetch papers relevant to the user interests: {user_input}")
    
    st.write(f'User input: {llm_response.content}')

    ## Parse the text generated by LLM into a list or dictionary of author-institution pairs
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

        # user_choice is used here to fetch the data from the database or API
        ## We need information about the author from some public API
        ## Institution information can be fetched from the database

        st.write("Thank you for using the web app!")


if __name__ == "__main__":
    main()

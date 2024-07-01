from dotenv import load_dotenv
import os
import google.generativeai as genai
import pandas as pd
import streamlit as st
from contextlib import redirect_stdout
from io import StringIO

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("api_key"))

# Initialize the model
model = genai.GenerativeModel("gemini-pro")

def answer_gemini(question):
    """Get answer from Gemini model."""
    response = model.generate_content(question)
    return response.text

def display_code_output(exec_code):
    """Execute and display code output."""
    with StringIO() as output_buffer:
        with redirect_stdout(output_buffer):
            try:
                exec(exec_code, globals())
            except Exception as e:
                return f"An error occurred: {e}"
        return output_buffer.getvalue()

# Streamlit app
st.title("Gemini Data Analyzer")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Top 10 rows of dataframe")
        st.write(df.head(10))

        # Identify categorical and numerical variables
        cat_var = [col for col in df.columns if df[col].dtype == 'object']
        num_var = [col for col in df.columns if df[col].dtype != 'object']

        if cat_var and num_var:
            question = (
                f"Use the dataframe df with columns {df.columns.tolist()} and tell me the feature selection techniques "
                f"I could use for this dataframe with {len(num_var)} numerical variables and {len(cat_var)} categorical variables. "
                f"Give me a few suggestions on what will be best."
            )

            with st.spinner("Getting recommendations..."):
                response = answer_gemini(question)
                st.subheader("Gemini Model Response")
                st.write(response)  # Display the response from the Gemini model

            # Extract and execute the code snippet from the response
            try:
                start_index = response.find('#')
                end_index = response.rfind(')')
                exec_code = response[start_index:end_index + 1]
                captured_output = display_code_output(exec_code)
                st.subheader("Captured Output:")
                st.code(captured_output, language='python')
            except Exception as e:
                st.error(f"Error in parsing or executing code: {e}")
        else:
            st.warning("The CSV should contain both numerical and categorical variables.")
        
        st.subheader("Ask Further Questions")
        additional_query = st.text_input("Enter your query about the dataframe:")

        if additional_query:
            with st.spinner("Getting response..."):
                further_response = answer_gemini(additional_query)
                st.subheader("Gemini Model Additional Response")
                st.write(further_response)
                # Extract and execute the code snippet from the additional response if any
                try:
                    start_index = further_response.find('#')
                    end_index = further_response.rfind(')')
                    if start_index != -1 and end_index != -1:
                        exec_code = further_response[start_index:end_index + 1]
                        additional_captured_output = display_code_output(exec_code)
                        st.subheader("Captured Output from Additional Query:")
                        st.code(additional_captured_output, language='python')
                except Exception as e:
                    st.error(f"Error in parsing or executing additional query code: {e}")
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
else:
    st.warning("Please upload a CSV file to proceed.")

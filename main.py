from dotenv import load_dotenv
import os
import google.generativeai as genai
import pandas as pd
import streamlit as st
from contextlib import redirect_stdout
from io import StringIO
import numpy as np

load_dotenv()
genai.configure(api_key=os.getenv("api_key"))

model = genai.GenerativeModel("gemini-pro")

def answer_gemini(question): #takes quesiton as prompt and answers
    response = model.generate_content(question)
    return response.text

st.title("Gemini Data Analayzer")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    #check if file exists
    df = pd.read_csv(uploaded_file)

    st.subheader("Top 10 rows of dataframe")
    st.write(df.head(10))

    #variable = st.text_input("Enter the prompt: ")
    cat_var = [col for col in df.columns if df[col].dtype == 'object']
    num_var = [col for col in df.columns if df[col].dtype != 'object']


    if cat_var and num_var:
        question = f"Use the dataframe df with columns {df.columns} and tell me the feature selection techniques I could use for this dataframe with {len(num_var)} numerical variables and {len(cat_var)} categorical varaibles. Give me a few suggestions on what will be best. " 

        response = answer_gemini(question)


        start_index1 = response.find('#')
        start_index2 = response.rfind(')')
        exec_code = response[start_index1:start_index2 + 1]

        with StringIO() as output_buffer:
            with redirect_stdout(output_buffer):
                try:
                    exec(exec_code)
                except Exception as e:
                    st.error(f"An error has occured: {e}")
            captured_output = output_buffer.getvalue()
        st.subheader("Captured Output:")
        st.code(captured_output, language='python')
else:
    st.warning("Please upload a CSV file to proceed")






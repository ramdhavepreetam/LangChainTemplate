## Integrate our code with OpenAI API

import os 
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
import streamlit as st


os.environ["OPENAI_API_KEY"] = openai_key
# streamlit framework 

st.title('Langchain Demo With OPENAI API')

input_text = st.text_input("Search the topic you want.")

#prompt templates 

first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about {name}"

)

llm=OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))


## OpenIA LLM Model setup. 


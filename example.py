## Integrate our code with OpenAI API

import os 
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory

import streamlit as st


os.environ["OPENAI_API_KEY"] = openai_key
# streamlit framework 

st.title('Langchain Demo With OPENAI API')
input_text = st.text_input("Search the topic you want.")

#prompt templates 
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template="Tell me about celebrity {name}"

)

# Memory 

person = ConversationBufferMemory(input_key='person', memory_key='person')
person_dob = ConversationBufferMemory(input_key='dob', memory_key='dob')
person_event = ConversationBufferMemory(input_key='event', memory_key='event')

llm=OpenAI(temperature=0.8)
chain = LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory= person)

#prompt templates 
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template="when was {person} born"

)

chain2 = LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=person_dob)

third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template="What measure events happned on aroud {dob} in the world"

)

chain3 = LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='event',memory=person_event)

parentchaing = SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','event'],verbose=True)

if input_text:
    st.write(parentchaing({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person.buffer)
    
    with st.expander('Events happnen'):
        st.info(person_event.buffer)



## OpenIA LLM Model setup. 


import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg"
import streamlit as st
import shutil
print(shutil.which("ffplay"))
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from sarvamai import SarvamAI
from sarvamai.play import play


load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro",
                            temperature=0.2,
                        )

client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
def extract_text(file_list):
    text=""
    for file in file_list:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200                                       
    )       

    chunks=text_splitter.split_text(text) 
    document=[]
    for chunk in chunks:
        document.append(Document(page_content=chunk)) 
             

#     [
#   Document(page_content="Clause 1: ABC"),
#   Document(page_content="Clause 2: DEF")
#     ]
#    
    return document

def translate_text(text):
    response = client.text.translate(
    input=text,
    source_language_code="auto",
    target_language_code="hi-IN",
    speaker_gender="Male"
)
    return response.translated_text

def speech(text):
    response = client.text_to_speech.convert(
    text=translated_text,
    target_language_code="hi-IN",
    enable_preprocessing=True
)

    play(response)
        
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template= """You are a knowledgeable assistant specializing in simplifying legal documents for users. Your tasks include:  

    1. **Plain-Language Summary:** Provide a clear, concise summary of the legal document in plain, easy-to-understand language.  

    2. **Highlighted Key Clauses:** Identify and clearly explain the key clauses, such as obligations, rights, penalties, and important dates.  

    3. **Warnings and Red Flags:** Highlight any potential risks, ambiguities, or concerning clauses that the user should be aware of.  

    Document:  
    {text}

    Response Format:  
    1. Plain-Language Summary:   

    2. Warnings and Red Flags:  
    """
)

summarized_chain=load_summarize_chain(
    llm=llm,
    chain_type="stuff",
    prompt=summary_prompt,
    verbose=True
)
    
st.title("DocEase: Simplifying Legal Documents with AI")
st.write("Upload your legal documents in PDF format, and let our AI assistant simplify them for you.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf", "png", "jpeg"],

    accept_multiple_files=True
)



button = st.button("Summarize")
if button:



    if uploaded_files:
        summary = extract_text(uploaded_files)
        summarized_text = summarized_chain.run(summary)
        shortsummarized_text = summarized_text[:1000]

        st.write("Summary:")
        st.write(shortsummarized_text)

        translated_text = translate_text(shortsummarized_text)
        st.write("Translated Summary:")
        st.write(translated_text)

        speech(translated_text)
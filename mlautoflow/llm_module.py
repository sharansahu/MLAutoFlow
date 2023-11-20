from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
import textwrap
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]

def initialize():
    global openai_llm, documents, texts, openai_embeddings, vectordb, retriever

    openai_llm = OpenAI(temperature=0.7, top_p=0.9, max_tokens = 1950)
    DOCUMENT_PATH = os.path.join(os.getcwd(), "output.txt")

    loader = TextLoader(DOCUMENT_PATH)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(document)

    openai_embeddings = OpenAIEmbeddings()
    persist_directory = 'db'

    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=openai_embeddings,
                                     persist_directory=persist_directory)

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(llm=openai_llm,
                                           chain_type="stuff",
                                           retriever=retriever)
    return qa_chain

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    return '\n'.join(wrapped_lines)

def handle_query(query):
    qa_chain = initialize()
    llm_response = qa_chain(query)
    response_str = wrap_text_preserve_newlines(llm_response['result'])
    return response_str

def extract_context(doc_path):
    with open(doc_path, 'r') as file:
        context = file.read()
    return context

def generate_predictor_code(context):
    prompt = f"""
    # Context: The information you have is from a Python project that uses PyTorch for image processing. The project includes functions for preprocessing and postprocessing images, as well as a trained PyTorch model for image analysis.

# Existing File Structure:
{context}

# Task: Based on the patterns and functions in the existing code and the file structure, complete the implementation of the Predictor class by filling in [Fill in based on repository code] and importing any necessary dependencies. Ensure to use appropriate functions from the codebase and include necessary imports. Do not create new methods that are not present in the provided code. 
# If you create functions, create them in this file. Do not response with "I don't know". Just try the best you can given the information provided. 
    """
    
    generated_code = handle_query(prompt)

    return generated_code

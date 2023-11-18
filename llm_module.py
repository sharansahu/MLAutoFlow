from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
import textwrap
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]

def initialize():
    global openai_llm, documents, texts, openai_embeddings, vectordb, retriever

    openai_llm = OpenAI()
    DOCUMENT_PATH = os.path.join(os.getcwd(), "output.txt")

    loader = PyPDFLoader()
    document = loader.load(DOCUMENT_PATH)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(document)

    openai_embeddings = OpenAIEmbeddings()
    persist_directory = 'db'

    vectordb = Chroma.from_documents(documents=[texts],
                                     embedding=openai_embeddings,
                                     persist_directory=persist_directory)

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=openai_llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True)
    return qa_chain

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    return '\n'.join(wrapped_lines)

def process_llm_response(llm_response):
    response_str = wrap_text_preserve_newlines(llm_response['result'])
    response_str += '\n\nSources:\n'
    for source in llm_response["source_documents"]:
        response_str += source.metadata['source'] + '\n'
    return response_str

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
    # Given the following code extracted from a repository:
    {context}

    # Complete the implementation of the following Predictor class based on the patterns and functions identified in the code above.

    from cog import BasePredictor, Input, Path

    class Predictor(BasePredictor):
        def setup(self) -> None:
            \"\"\"Load the model into memory to make running multiple predictions efficient\"\"\"
            # [Fill in based on repository code]

        def predict(
            self,
            image: Path = Input(description="Grayscale input image"),
            scale: float = Input(
                description="Factor to scale image by", ge=0, le=10, default=1.5
            ),
        ) -> Path:
            \"\"\"Run a single prediction on the model\"\"\"
            # [Fill in based on repository code]
    """

    # Use the LLM to generate the code
    generated_code = handle_query(prompt)

    # Integrate the generated code into the Predictor class template
    return generated_code


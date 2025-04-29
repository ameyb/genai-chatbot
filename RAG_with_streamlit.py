#Import Library
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda
from langchain.globals import set_debug
from langchain_core.tracers.stdout import ConsoleCallbackHandler

from langchain_postgres.vectorstores import PGVector
from database import COLLECTION_NAME, CONNECTION_STRING
# from langchain_community.storage import RedisStore
from cassandra.cluster import Cluster
from langchain_community.storage import CassandraByteStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from pathlib import Path
from IPython.display import display, HTML
from base64 import b64decode
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stateful_chat import chat as fancy_chat

import os, hashlib, shutil, uuid, json, time
import torch, streamlit as st
import logging


from dotenv import load_dotenv
load_dotenv()
set_debug(True)
# Ensure PyTorch module path is correctly set
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Redis client
# client = redis.Redis(host="localhost", port=6379, db=0)

# Initialize YCQL client
cluster = Cluster(['10.150.2.56'])
session = cluster.connect()

# Create the keyspace.
session.execute('CREATE KEYSPACE IF NOT EXISTS langchain;')

# Create a YCQL table 
session.execute(
  """
  CREATE TABLE IF NOT EXISTS langchain.pdf_hash_store (pdf_hash varchar PRIMARY KEY,
                                              status varchar);
  """)

# Global Retriever variable for storing and searching documents
retriever_with_byteStore_vectorStore = None

# Global variable to store log data
# Initialize log data
log_data = f"\n\nApp started"
log_area = None
def log_data_to_ui(new_log):
    global log_data
    global log_area
    log_data += new_log
    if log_area is not None:  # <- ✅ Protect against None
        log_area.markdown(f"```\n{log_data}\n```")
    

#Data Loading
def load_pdf_data(file_path):
    logging.info(f"Data ready to be partitioned and loaded ")
    raw_pdf_elements = partition_pdf(
        filename=file_path,
      
        infer_table_structure=True,
        strategy = "hi_res",
        
        extract_image_block_types = ["Image"],
        extract_image_block_to_payload  = True,

        chunking_strategy="by_title",     
        mode='elements',
        max_characters=10000,
        new_after_n_chars=5000,
        combine_text_under_n_chars=2000,
        image_output_dir_path="data/",
    )
    logging.info(f"Pdf data finish loading, chunks now available!")
    return raw_pdf_elements

# Generate a unique hash for a PDF file
def get_pdf_hash(pdf_path):
    """Generate a SHA-256 hash of the PDF file content."""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return hashlib.sha256(pdf_bytes).hexdigest()

# Summarize extracted text and tables using LLM
def summarize_text_and_tables(text, tables):
    logging.info("Ready to summarize data with LLM")
    prompt_text = """You are an assistant tasked with summarizing text and tables. \
    
                    You are to give a concise summary of the table or text and do nothing else. 
                    Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0.6, model="gpt-4o-mini", callbacks=[ConsoleCallbackHandler()])
    summarize_chain = {"element": RunnablePassthrough()}| prompt | model | StrOutputParser()
    logging.info(f"{model} done with summarization")
    return {
        "text": summarize_chain.batch(text, {"max_concurrency": 5}),
        "table": summarize_chain.batch(tables, {"max_concurrency": 5})
    }
  
#Initialize a pgvector and retriever for storing and searching documents
def initialize_retriever():

    # store = RedisStore(client=client)
 
    store = CassandraByteStore(
                    table="pdf_byte_store",
                    session=session,
                    keyspace="langchain",
                )
    id_key = "doc_id"
    vectorstore = PGVector(
            embeddings=OpenAIEmbeddings(),
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
            )
    retrieval_loader = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
    return retrieval_loader


# Store text, tables, and their summaries in the retriever

def store_docs_in_retriever(text, text_summary, table, table_summary, retriever):
    """Store text and table documents along with their summaries in the retriever."""

    def add_documents_to_retriever(documents, summaries, retriever, id_key = "doc_id"):
        """Helper function to add documents and their summaries to the retriever."""
        if not summaries:
            return None, []

        doc_ids = [str(uuid.uuid4()) for _ in documents]
        summary_docs = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]

        # Convert documents to byte format
        byte_documents = [doc.encode("utf-8") if isinstance(doc, str) else doc for doc in documents]

        retriever.vectorstore.add_documents(summary_docs, ids=doc_ids)
        retriever.docstore.mset(list(zip(doc_ids, byte_documents)))

    # Add text, table, and image summaries to the retriever
    add_documents_to_retriever(text, text_summary, retriever)
    add_documents_to_retriever(table, table_summary, retriever)
    
    return retriever


# Parse the retriever output
def parse_retriver_output(data):
    parsed_elements = []
    for element in data:
        # Decode bytes to string if necessary
        if isinstance(element, bytes):
            element = element.decode("utf-8")
        
        parsed_elements.append(element)
    
    return parsed_elements


# Chat with the LLM using retrieved context
def chat_with_llm(retriever, previous_context=None):
    logging.info(f"Context ready to send to LLM ")


    prompt_text = """Human: You are an International Trade and Tariff Regulations expert who can answer questions only based on the context provided below.

                    Answer the question STRICTLY based on the US Tariff data context provided in JSON below.

                    Do not assume or retrieve any information outside of the context.

                    Keep the answer concise, using three sentences maximum for text responses.

                    Think step-by-step to ensure accuracy based on the context provided.

                    If multiple results exist (e.g., multiple tariff rates, product categories, exemptions), present them as a bulleted list, numbered list, or in a table.

                    If numerical comparisons or trends are evident in the context, you may present a simple chart or table to make the information easier to understand.

                    Use tables for structured data (e.g., tariff codes, product descriptions, rates) and simple bar or line charts for comparisons (e.g., tariff rate trends).

                    Do not add any extra commentary, apologies, summaries, or retrievals beyond the provided context.

                    Do not start the response with "Here is a summary" or similar phrasing.

                    If the context is empty, respond with: None.

                    Always prioritize clarity and relevance in the format you choose (plain text, table, or chart).

                    Here is the context:
                    <context>
                    {context}
                    </context>

                    Question:
                    {question}
                """





# Combine previous context with the new context
    def combine_contexts(new_context, previous_context):
        if previous_context:
            return f"{previous_context}\n\n{new_context}"
        return new_context

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")
 
    rag_chain = ({
                    "context": retriever | RunnableLambda(
                            lambda output: combine_contexts(
                                            "\n".join(parse_retriver_output(output) if isinstance(output, list) else []),
                                            previous_context
                                        )
                            ),
                    "question": RunnablePassthrough(),
                } 
                        | prompt 
                        | model 
                        | StrOutputParser()
            )
    
    print(rag_chain)
    logging.info(f"Completed! ")
    return rag_chain

# Generate temporary file path of uploaded docs
def _get_file_path(file_upload):

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists

    if isinstance(file_upload, str):
        file_path = file_upload  # Already a string path
    else:
        file_path = os.path.join(temp_dir, file_upload.name)
        with open(file_path, "wb") as f:
            f.write(file_upload.getbuffer())
        return file_path
    

# Process uploaded PDF file
def process_pdf(file_upload):
    print('Processing PDF hash info...')
    log_data_to_ui("\n\nProcessing PDF hash info...")
    global retriever_with_byteStore_vectorStore
    retriever_with_byteStore_vectorStore = None
    if retriever_with_byteStore_vectorStore is None:
        retriever_with_byteStore_vectorStore = initialize_retriever()
        print("Retriever initialized")
        log_data_to_ui("\n\nRetriever initialized")
        
    for file in file_upload:    
        
        print("Processing file: {}".format(file))
        log_data_to_ui("\n\nProcessing file: {}".format(file.name))
        file_path =  _get_file_path(file)
        pdf_hash = get_pdf_hash(file_path)
        print("\n\nPDF hash generated for file {}".format(file.name))
    
        query = f"SELECT COUNT(*) FROM langchain.pdf_hash_store WHERE pdf_hash='{pdf_hash}';"
        existing = session.execute(query).one()[0] > 0
        
        if existing:
            print(f"PDF already exists with hash {pdf_hash}. Skipping upload.")
            log_data_to_ui(f"\n\nPDF already exists with hash {pdf_hash}. Skipping upload.")
            return retriever_with_byteStore_vectorStore


        print(f"New PDF detected. Processing... {pdf_hash}")
        log_data_to_ui(f"\n\nNew PDF detected. Processing... {pdf_hash}")
 
        pdf_elements = load_pdf_data(file_path)
    
        tables = [element.metadata.text_as_html for element in
                pdf_elements if 'Table' in str(type(element))]
    
        text = [element.text for element in pdf_elements if 
                'CompositeElement' in str(type(element))]
   

        summaries = summarize_text_and_tables(text, tables)
        retriever_with_byteStore_vectorStore = store_docs_in_retriever(
            text, summaries['text'], tables,  summaries['table'], 
            retriever_with_byteStore_vectorStore)
        log_data_to_ui(f"\n\nSummarized data and stored in retriever")
        
        # Insert the PDF hash into the YCQL table
        
        session.execute(
            f"""
            INSERT INTO langchain.pdf_hash_store (pdf_hash, status)
            VALUES ('{pdf_hash}', '{json.dumps({"text": "PDF processed"})}');
            """
        )
        
        query = f"SELECT COUNT(*) FROM langchain.pdf_hash_store WHERE pdf_hash='{pdf_hash}';"
        stored = session.execute(query).one()[0] > 0
        
        print(f"Stored PDF hash in YB YCQL: {'Success' if stored else 'Failed'}")
        log_data_to_ui(f"\n\nStored PDF hash in YB YCQL: {'Success' if stored else 'Failed'}")
        
    return retriever_with_byteStore_vectorStore


#Invoke chat with LLM based on uploaded PDF and user query
def invoke_chat(file_upload, message, previous_context=None):
    retriever = process_pdf(file_upload)
    log_data_to_ui(f"\n\nGenerate Prompt with Context and User Query")
    rag_chain = chat_with_llm(retriever, previous_context)
    response = rag_chain.invoke(message)
    log_data_to_ui(f"\n\nChat with LLM invoked")
    response_placeholder = st.empty()
    response_placeholder.write(response)
    return response

# Main application interface using Streamlit
def main():
  
    st.set_page_config(
        page_title="Economic Assistant",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <div style="text-align: center; padding: 20px; background-image: url('data:image/jpeg;base64,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');">
            <h1 style='font-size: 50px; color: #FFB347;'>Economic Assistant</h1>
            <h4 style='color: white;'>Understanding Trade Impacts, One Document at a Time</h4>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    
    chatBotCol, loggerCol = st.columns([1, 1], gap="large")
    global log_area
    with loggerCol:
        log_area = st.empty()
    
     # Sidebar: File Upload + Sample Questions
    with st.sidebar:
        st.header("📦 Import Trade Documents")
        file_upload = st.file_uploader(
            label="Upload your Trade PDFs",  
            type=["pdf"], 
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if file_upload:
            st.success(f"{len(file_upload)} file(s) uploaded!", icon="✅")
            st.write(f"**Files Uploaded:**")
            for file in file_upload:
                st.write(f"- {file.name}")
        else:
            st.info("📂 Please upload one or more tariff documents to begin.")

        st.markdown("---")
        st.subheader("🔎 Try Asking:")
        sample_questions = "What are the latest aluminum tariffs?", "Steel import tax from China?"

        for text in sample_questions:
            if st.button(text, key=text):
                st.session_state["sample"] = text

    # Chat Section
    with chatBotCol:
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'context' not in st.session_state:
            st.session_state.context = None

        add_vertical_space(1)

        # stylable_container(
        #     key="chat_container",
        #     css_styles="""
        #         {
        #             background-color: #f4f6f8;
        #             padding: 20px;
        #             border-radius: 12px;
        #             box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        #         }
        #     """,
        # )

        # Fancy chat input
        user_prompt = st.chat_input("Ask a question about the uploaded document...")


        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            st.session_state["sample"] = None

        if "sample" in st.session_state and st.session_state["sample"] is not None:
            user_input = st.session_state["sample"]
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state["sample"] = None

        # Show chat history nicely
        for idx, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"**You:** {message['content']}")
            elif message["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(f"**Assistant:** {message['content']}")

        # Generate assistant response
        if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                start_time = time.time()

                with st.spinner("✨ Thinking..."):
                    latest_message = st.session_state.messages[-1]
                    user_message = latest_message["content"]
                    response_message = invoke_chat(file_upload, user_message)

                    duration = time.time() - start_time
                    response_content = f"{response_message}\n\n⏱️ *Response Time: {duration:.2f} seconds*"
                    st.markdown(response_content)

                    st.session_state.messages.append({"role": "assistant", "content": response_content})

    # Logger Section
    with loggerCol:
        with st.expander("📜 Logs (click to expand)", expanded=False):
            log_area.markdown(f"```\n{log_data}\n```")
    
    logging.info("App started")

   
if __name__ == "__main__":
    main()
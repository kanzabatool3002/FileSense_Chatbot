import os 
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from docx import Document

def get_single_pdf_chunks(pdf, text_splitter):
    pdf_reader = PdfReader(pdf)
    pdf_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        page_chunks = text_splitter.split_text(page_text)
        pdf_chunks.extend(page_chunks)
    return pdf_chunks
def get_single_docx_chunks(docx_file, text_splitter):
    doc = Document(docx_file)
    doc_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text_splitter.split_text(doc_text)


def get_all_pdfs_chunks(files):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    
    all_chunks = []
    for file in files:
        if file.name.endswith(".pdf"):
            pdf_chunks = get_single_pdf_chunks(file, text_splitter)
            all_chunks.extend(pdf_chunks)
        elif file.name.endswith(".docx"):
            docx_chunks = get_single_docx_chunks(file, text_splitter)
            all_chunks.extend(docx_chunks)
    return all_chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except:
        st.warning("Issue with reading the PDF/s. Your File might be scanned so there will be nothing in chunks for embeddings to work on")

def get_response(context, question, model):

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    prompt_template = f"""
    You are a helpful and informative bot that answers questions using text from the reference context included below. \
    Be sure to respond in a complete sentence, providing in depth, in detail information and including all relevant background information. \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    strike a friendly and converstional tone. \
    If the passage is irrelevant to the answer, you may ignore it also as a Note: Based on User query Try to Look into your chat History as well
    
    Context: {context}?\n
    Question: {question}\n
    """

    try:
        response = st.session_state.chat_session.send_message(prompt_template)
        return response.text

    except Exception as e:
        st.warning(e)

def working_process(generation_config):

    system_instruction = "You are a helpful document answering assistant. You care about user and user experience. You always make sure to fulfill user request"

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction)

    vectorstore = st.session_state['vectorstore']

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a FileSense Assistant. Ask me anything about your Files")
    ]
    
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Enter Your Query....")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)
        
        with st.chat_message("AI"):
            try:
                relevant_content = vectorstore.similarity_search(user_query, k=10)
                result = get_response(relevant_content, user_query, model)
                st.markdown(result)
                st.session_state.chat_history.append(AIMessage(content=result))
            except Exception as e:
                st.warning(e)


def main():
    load_dotenv()

    st.set_page_config(page_title="FileSense Chatbot", page_icon=":books:")
    st.header("FileSense Chatbot :books:")

    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 8000,
    }

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []

    # Display uploader in the center if no PDFs are uploaded
    if not st.session_state.pdf_uploaded:
        st.subheader("Upload your Documents to Start")
        pdf_docs = st.file_uploader("Upload your Documents here", accept_multiple_files=True)

        if st.button("Submit"):
            if pdf_docs:
                with st.spinner("Processing Documents..."):
                    st.session_state.pdf_docs.extend(pdf_docs)  # Store PDFs in session state
                    text_chunks = get_all_pdfs_chunks(st.session_state.pdf_docs)
                    vectorstore = get_vector_store(text_chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_uploaded = True  # Set flag to move uploader to sidebar
                    st.rerun()  # Refresh the page to move uploader to sidebar

    # Sidebar for uploading additional PDFs and managing existing ones
    if st.session_state.pdf_uploaded:
        with st.sidebar:
            st.subheader("Your Documents")
            st.success(f"{len(st.session_state.pdf_docs)} Documents Uploaded Successfully!")


            # Display the list of uploaded PDFs with remove buttons
            st.subheader("Uploaded Documents:")
            for i, pdf in enumerate(st.session_state.pdf_docs):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(pdf.name)  # Show PDF name
                with col2:
                    if st.button(f"X", key=f"remove_{i}"):
                        st.session_state.pdf_docs.pop(i)  # Remove selected PDF
                        if st.session_state.pdf_docs:
                            text_chunks = get_all_pdfs_chunks(st.session_state.pdf_docs)
                            vectorstore = get_vector_store(text_chunks)
                            st.session_state.vectorstore = vectorstore
                        else:
                            st.session_state.vectorstore = None
                            st.session_state.pdf_uploaded = False  # Reset if no PDFs left
                        st.rerun()  # Refresh the page


            # Option to add more PDFs
            additional_pdfs = st.file_uploader("Add more Documents", accept_multiple_files=True)
            if st.button("Add Documents"):
                if additional_pdfs:
                    with st.spinner("Processing new Documents..."):
                        st.session_state.pdf_docs.extend(additional_pdfs)
                        text_chunks = get_all_pdfs_chunks(st.session_state.pdf_docs)
                        vectorstore = get_vector_store(text_chunks)
                        st.session_state.vectorstore = vectorstore
                        st.rerun()

            if st.button("Reset & Upload New Documents"):
                st.session_state.pdf_docs = []
                st.session_state.vectorstore = None
                st.session_state.pdf_uploaded = False
                st.rerun()

    if st.session_state.vectorstore is not None:
        working_process(generation_config)

if __name__ == "__main__":
    main()

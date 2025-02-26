# FileSense Chatbot

## Overview
FileSense Chatbot is an AI-powered document assistant that allows users to upload multiple PDF and DOCX files and interact with them through natural language queries. Using **Retrieval-Augmented Generation (RAG)**, the chatbot retrieves relevant information from uploaded documents and generates coherent responses. It leverages Google's Generative AI, **LangChain**, and FAISS for efficient vector-based search and response generation.

## Features
- Supports uploading multiple **PDF** and **DOCX** files
- Extracts and processes text from documents
- Uses **LangChain** for text processing and vector storage
- Utilizes **FAISS** for document retrieval
- Leverages **Google Generative AI** for response generation
- Maintains chat history for contextual responses
- Provides a user-friendly interface using **Streamlit**

## **RAG (Retrieval-Augmented Generation)**
This project is built using **Retrieval-Augmented Generation (RAG)**, an advanced approach that combines retrieval and generation techniques to provide accurate and context-aware responses.

### **Key Steps in the RAG Process**
1. **Retrieval**: The chatbot stores text embeddings in a **vector store (FAISS)**, allowing it to efficiently retrieve relevant information based on user queries.
2. **Generation**: Using Google's **Generative AI model**, the chatbot processes the retrieved content and generates a well-formed, contextually relevant response.

By integrating retrieval with generative AI, FileSense Chatbot enhances response accuracy, making it an effective document search assistant.

## **LangChain Integration**
FileSense Chatbot leverages **LangChain** to enhance document processing and conversational AI capabilities. Key LangChain components used in this project include:

- **Text Splitting**: Uses `RecursiveCharacterTextSplitter` to split document text into manageable chunks for efficient retrieval.
- **Message Handling**: Implements `AIMessage` and `HumanMessage` to manage chat history and maintain context.
- **Vector Store Integration**: Utilizes `FAISS` from LangChain to store and retrieve document embeddings efficiently.
- **Google AI Embeddings**: Uses `GoogleGenerativeAIEmbeddings` to generate embeddings for document text, enabling better search and retrieval.

With LangChain, the chatbot ensures optimal processing of large documents and maintains smooth, intelligent interactions.

## How to Run It Locally
### **1. Clone the Repository**
```bash
git clone https://github.com/kanzabatool3002/FileSense_Chatbot.git
cd fileSense-chatbot
```

### **2. Set Up Environment Variables**
Create a `.streamlit` folder in the project directory and add a `secrets.toml` file:
```bash
mkdir .streamlit
nano .streamlit/secrets.toml
```
Add the following:
```toml
[secrets]
GOOGLE_API_KEY = "your_google_api_key"
```

### **3. Create a Virtual Environment and Install Dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### **4. Run the Application**
```bash
streamlit run app.py
```
Access the interface at: **[http://localhost:8501](http://localhost:8501)**

## Applications
FileSense Chatbot can be used in various domains:
- **Academic Research**: Quickly extract and retrieve information from research papers
- **Legal Documents**: Search for relevant sections in lengthy contracts and legal files
- **Corporate Documentation**: Retrieve policies, guidelines, or reports efficiently
- **Healthcare**: Summarize and query medical documents for patient care
- **Education**: Help students and teachers find relevant information in study materials

---
**FileSense Chatbot** - Making Document Search Smarter! 🚀


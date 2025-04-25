### 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for efficient document processing and knowledge retrieval. It extracts text and tables from PDFs using the **Unstructured** library, stores raw PDFs, and indexes extracted embeddings in **PGVector** for semantic search. The system leverages **MultiVector Retriever** for context retrieval before querying an **LLM (GPT model)**.

### 🚀 Features

- **Unstructured Document Processing**: Extracts text and tables from PDFs.  
- **YugabyteDB for Raw Storage**: Stores and retrieves raw PDFs efficiently, to implement persistent storage.  
- **YugabyteDB's PGVector support for Vector Storage**: Indexes and retrieves high-dimensional embeddings for similarity search.  
- **MultiVector Retriever**: Optimized for retrieving contextual information from multiple sources.  
- **LLM Integration**: Uses a **GPT model** or anyother LLM implementation to generate responses based on retrieved context.  

### 🛠️ Tech Stack

#### Programming Language
- Python  

#### Libraries
- `unstructured`
- `pgvector`
- `langchain`
- `openai`


#### Databases
- **YugabyteDB**: For raw PDF storage  
- **YugabyteDB + PGVector**: For embeddings storage  
YugabyteDB provides a scalable vector store and a globally consistent system of record, ideal for RAG (Retrieval-Augmented Generation) architectures that require fast, real-time access to relevant data.

#### Workflow Orchesration
LittleHorse.io handles the AI workflow orchestration, enabling scalable, reliable microservice coordination with built-in support for retries, state management, and observability. Together, they enable the next generation of context-aware, workflow-driven generative AI applications.

#### LLM
- **GPT** (via OpenAI API or local model)

Inspired by the work done done LangChain

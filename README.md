# Simple RAG with LangChain using Persian LLM

## 🚀 Overview

This project demonstrates a Retrieval Augmented Generation (RAG) system tailored for the Persian language. It leverages Persian language models to understand and answer questions based on information contained within a custom knowledge base of PDF documents. The primary goal is to enhance the capabilities of Large Language Models (LLMs) by grounding their responses in factual, externally provided context, thereby improving accuracy and reducing hallucinations.

This implementation uses `PartAI/Dorna2-Llama3.1-8B-Instruct` as the core Large Language Model for generation and `PartAI/Tooka-SBERT-V2-Large` for generating dense vector embeddings for efficient retrieval.

## ✨ Key Features

* **Persian Language Focus:** Specifically designed and implemented using models optimized for Persian.
* **PDF Document Processing:** Ingests and processes text from PDF files to build the knowledge base.
* **State-of-the-Art Persian Models:** Utilizes `Dorna2-Llama3.1-8B-Instruct` for generation and `Tooka-SBERT-V2-Large` for embeddings.
* **LangChain Framework:** Orchestrates the entire RAG pipeline, from data ingestion to answer generation.
* **Local Vector Store:** Employs a local vector database (e.g., ChromaDB or FAISS) for storing and querying document embeddings.

## 🛠️ Models & Technology Stack

* **Core LLM (Generator):**
    * **Model:** `PartAI/Dorna2-Llama3.1-8B-Instruct`
    * **Description:** A powerful instruction-tuned Llama 3.1 based model with 8 billion parameters, specifically adapted and optimized for the Persian language by Part AI.
* **Embedding Model (Retriever):**
    * **Model:** `PartAI/Tooka-SBERT-V2-Large`
    * **Description:** A large SBERT-based model by Part AI, designed to generate high-quality sentence and document embeddings for Persian text, enabling effective semantic search.
* **Key Libraries & Frameworks:**
    * Python 3.x
    * LangChain
    * Hugging Face Transformers (for model loading and pipelines)
    * Sentence Transformers (often used with SBERT models, or wrapped by `HuggingFaceEmbeddings`)
    * ChromaDB / FAISS (for local vector storage)
    * PyPDF (for loading PDF documents)
    * PyTorch

## ⚙️ Implementation Overview

The RAG pipeline was implemented as follows:

1.  **Document Loading:**
    * Persian PDF documents serving as the knowledge base are loaded using `PyPDFLoader` from LangChain.
2.  **Text Chunking:**
    * The loaded text is split into smaller, manageable, and semantically coherent chunks using `RecursiveCharacterTextSplitter`. This ensures that the chunks fit within the context window of the models and provide focused context.
3.  **Embedding Generation:**
    * The `PartAI/Tooka-SBERT-V2-Large` model is used via `HuggingFaceEmbeddings` to generate dense vector embeddings for each text chunk. These embeddings capture the semantic meaning of the chunks.
4.  **Vector Storage & Indexing:**
    * The generated embeddings and their corresponding text chunks are stored and indexed in a local vector database (e.g., ChromaDB). This allows for efficient similarity searches.
5.  **User Query & Retrieval:**
    * When a user poses a question in Persian, the query is also embedded using the `PartAI/Tooka-SBERT-V2-Large` model.
    * The vector store then performs a similarity search to find and retrieve the most relevant text chunks from the knowledge base based on the query embedding.
6.  **Prompt Engineering & Context Augmentation:**
    * The retrieved text chunks (context) are combined with the original user query and a carefully crafted prompt.
    * This prompt is designed to instruct the `PartAI/Dorna2-Llama3.1-8B-Instruct` LLM on how to use the provided context to answer the question accurately in Persian, adhering to any specific formatting or constraints (e.g., Llama 3's chat template structure with system, user, and assistant roles).
7.  **Answer Generation:**
    * The augmented prompt (context + query + instructions) is fed to the `PartAI/Dorna2-Llama3.1-8B-Instruct` model.
    * The LLM generates a response in Persian, grounded in the information retrieved from the PDF documents.
8.  **Orchestration:**
    * The entire workflow is orchestrated using the LangChain framework, connecting the different components (loaders, splitters, embedding model, vector store, retriever, and LLM) into a cohesive RAG chain (e.g., `RetrievalQA`).

## 📊 Sample Outputs & Results

**Example 1:**

* **Question:** `داور فینال لیگ قهرمانان اروپا 2026 کی بود؟`
* **Answer:** `آقای کلمن تورپن، داور فرانسوی، مسئولیت هدایت بازی فینال لیگ قهرمانان اروپا در سال ۲۰۲۶ را بر عهده داشت.`

---

**Example 2:**

* **Question (Persian):** `دقیقه 67 رو توصیف کن`
* **Answer (Persian):** `دقیقه ۶۷: بازی به لحظه‌های حساس و پرتنش نزدیک می‌شود. هر دو تیم با شدت تلاش می‌کنند تا گل برتری را به ثمر بنشانند و برتری خود را در بازی نشان دهند. این لحظه‌ها معمولاً پر از هیجان و درخشش است و بازیکنان با تلاش‌های شدید خود، روحیه و انرژی خود را به حداکثر می‌رسانند.`

---

**Example 3:**

* **Question (Persian):** `بارسلونا چند گل زد؟`
* **Answer (Persian):** `بارسلونا ۲ گل زد.`

---

**Example 4:**

* **Question (Persian):** `در مورد لامین یامال بهم بگو`
* **Answer (Persian):** `لامینه یامال، یک بازیکن فوتبال ۱۸ ساله است که برای باشگاه بارسلونا بازی می‌کند. او در بازی اخیر تیمش، گل پیروزی را به ثمر رساند و همچنین در دقیقه ۵۶، با یک حرکت انفرادی خارقالعاده، گل تساوی را نیز به ثمر رساند. این گل‌های او در دقایق پایانی بازی، سرنوشت بازی را تغییر داد و باعث شد تیمش پیروز شود.`

---

## 🔮 Future Improvements

* Integration of more advanced retrieval strategies (e.g., hybrid search, re-ranking with a cross-encoder).
* Fine-tuning `PartAI/Dorna2-Llama3.1-8B-Instruct` further on domain-specific Persian question-answering datasets.
* Developing a user interface for easier interaction with the RAG system.
* Implementing robust evaluation metrics to systematically track performance.

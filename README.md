# Career and Workplace Assistant

This repository contains a **Career and Workplace Assistant**, an AI-powered application that provides tailored advice and resources for users facing workplace challenges or seeking career transitions. Built using **Streamlit**, **LangChain**, **Twelve Labs**, **PyMilvus**, and **Google Generative AI**, this project demonstrates how AI can offer valuable guidance and support for professionals.

---

## Features

1. **Intent Classification**:
   - Uses Google Generative AI to classify user queries into two categories:
     - **Career Transition**: For learning paths, career development, and related guidance.
     - **Workplace Stress**: For dealing with stress, workplace behavior, and anxiety.

2. **Resource Retrieval**:
   - Retrieves the most relevant **PDFs** and **videos** based on the user query using:
     - **HuggingFace embeddings** for text-based resources.
     - **Twelve Labs embeddings** for video-based resources.
   - Indexed and queried through **PyMilvus** for efficient search.

3. **Interactive Interface**:
   - A user-friendly interface powered by **Streamlit**, with features like query input, clickable thumbnails, and structured responses.

4. **Thumbnail Previews**:
   - Displays thumbnails for PDFs and videos alongside clickable links.

5. **Custom Learning Paths**:
   - Provides structured advice using Google Generative AI, tailored to the user query.

---

## Technology Stack

- **Streamlit**: Interactive user interface.
- **LangChain**: For text splitting and embeddings.
- **Twelve Labs**: For video embeddings.
- **PyMilvus**: Vector database for storing and retrieving embeddings.
- **Google Generative AI**: For generating structured learning paths and advice.
- **HuggingFace Embeddings**: For text embeddings.
- **PyPDF2**: For extracting text from PDFs.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/career-workplace-assistant.git
   cd career-workplace-assistant

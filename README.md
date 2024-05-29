Studious.AI: AI-driven Educational Chatbot
> This project implements a question-answering system over uploaded PDFs using Chainlit. Users can upload a study material (PDF) and then ask questions about the content. The system retrieves relevant passages and provides answers based on a conversational model and a vector store for efficient information retrieval.

Features:
- Processes a single uploaded PDF and extracts text.
- Splits the extracted text into manageable chunks.
- Creates a vector store for efficient search using pre-trained sentence embeddings.
- Enables users to ask questions about the PDF in a conversational manner.
- Retrieves relevant passages from the PDF based on the user's questions.
- Provides answers that incorporate the retrieved passages for context.

Requirements:
- Python 3.10.11
- Chainlit
- PyPDF2 (https://pypdf2.readthedocs.io/)
- Langchain libraries (OllamaEmbeddings, ConversationalRetrievalChain, etc.)

Installation:
- Clone this repository.
- Install the required libraries using pip install -r requirements.txt.

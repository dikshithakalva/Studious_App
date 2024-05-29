import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
import serpapi


llm_local = ChatOllama(model="mistral:instruct")

@cl.on_chat_start
async def on_chat_start():

    files = None  # Initialize variable to store uploaded files

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload your study materials to begin!",
            accept=["application/pdf"],
            max_size_mb=100,
            timeout=180,
        ).send()

    file = files[0]  # Get the first uploaded file

    # Inform the user that processing has started
    msg = cl.Message(content=f"Processing your study material`{file.name}`...")
    await msg.send()

    # Read the PDF file
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_local,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    # Store the chain in user session
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):

    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")

    # Callbacks happen asynchronously/parallel
    cb = cl.AsyncLangchainCallbackHandler()

    # Call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []  # Initialize list to store text elements

    # Process source documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found in document"

    # Call summarize API if answer is short or not found in document
    if len(answer) < 50 or not source_documents:
        # Use web search to find answer if not in doc
        try:
            search_results = google_search(message.content)  # Replace with your preferred web search function
            answer = f"Answer not found in document. Here's what I found on the web: {search_results}"
        except Exception:
            answer = "Sorry, I couldn't find relevant information."

    # Return results
    await cl.Message(content=answer, elements=text_elements).send()

def google_search(query):
    params = {
        "api_key": "40104566cfe7c48f63a0377c9a1b24aa89004422a261bc7b3b7de1fbec7ff741",  # Replace with your actual API key
        "engine": "google",
        "q": query,
        "api_version": "v1"
    }

    search = serpapi.search(params)
    results = search.get_dict()

    if results.get("organic_results"):
        # Access and return top search result URL
        top_result = results["organic_results"][0]
        return f"Answer not found in document. Top search result: {top_result['link']}"
    else:
        return "Sorry, I couldn't find relevant information on the web."

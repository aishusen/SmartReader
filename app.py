import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import chainlit as cl 
from chainlit.types import AskFileResponse

# HuggingFace pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load the model locally (no API key needed)
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

welcome_message = """Welcome to Pluto.ai! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    loader = Loader(file.path)
    pages = loader.load()
    return pages

def split_into_chunks(file: AskFileResponse):
    pages = process_file(file)

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = character_splitter.split_documents(pages)
    
    for i, doc in enumerate(splits):
        doc.metadata["source"] = f"source_{i}"
            
    print(f"Number of chunks: {len(splits)}")
    return splits

def store_embeddings(chunks):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding_function)
    print(f"Size of vectordb: {vectordb._collection.count()}")
    return vectordb

@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Pluto",
        url="https://avatars.githubusercontent.com/u/128686189?s=400&v=4",
    ).send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    chunks = split_into_chunks(file)

    msg.content = f"Creating chunks for `{file.name}`..."
    await msg.update()

    vectordb = store_embeddings(chunks)

    msg.content = f"Creating embeddings for `{file.name}`..."
    await msg.update()

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a chat history and the latest user question, rewrite the question standalone if it depends on context. Otherwise return it as is."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant. Use the retrieved context to answer the question. If not enough information, say 'I don't know'.\n\n{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    response = await chain.ainvoke(
        {"input": message.content},
        config={"configurable": {"session_id": "abc123"}, "callbacks": [cl.AsyncLangchainCallbackHandler()]},
    )

    answer = response["answer"]
    source_documents = response["context"]
    text_elements = []
    unique_pages = set()

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            page_number = source_doc.metadata.get('page', source_idx)
            page = f"Page {page_number}"
            if page not in unique_pages:
                unique_pages.add(page)
                text_elements.append(cl.Text(content=source_doc.page_content, name=page))
        answer += f"\n\nSources: {', '.join(unique_pages)}" if unique_pages else "\n\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()

import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain.llms import HuggingFaceHub
# from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
# import torch

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:                  #loop through each pdf
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:     #go through each page in pdf
            text += page.extract_text()
    
    return text


def get_text_chunks(text):          #need to use Langchain text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 512,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    st.write("Embedding text chunks into vectors..")
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl", 
                                               model_kwargs = {"device": "cuda"})
    st.write("Storing embedded vectors into vector database...")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs={"temperature" : 0.6, "max_length" : 1024})

    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages= True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )

    return conversation_chain

def handle_user_qns(user_qns):
    response = st.session_state.conversation({'question': user_qns})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    
    # load_dotenv()     #access the env file containing all the tokens and keys
    st.set_page_config(page_title="Ask PDFs")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:           #to initialize it to an empty list, so that everytime when we have conversation, it gets appended to the list. This is to prevent overriding of messages.
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ask PDFs")
    user_qns = st.text_input("Ask a question about your documents:")
    if user_qns:
        handle_user_qns(user_qns)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files= True)
        if st.button("Process"):
            with st.spinner("Processing"):
                torch.cuda.empty_cache()

                #get pdf text 
                st.write("Getting text data from PDF documents...")
                raw_text = get_pdf_text(pdf_docs)

                #get the text chunks
                st.write("Splitting the raw text into chunk size of 512...")
                text_chunks = get_text_chunks(raw_text)

                #create vector store
                vector_store = get_vectorstore(text_chunks)
                st.write("Completed!")

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)      #need to include st.session state because streamlit has a tendency to reload the code, so in order for the conversation to be persistent, we need to use this code. 

if __name__=='__main__':
    main()
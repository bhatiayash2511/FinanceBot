import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv('.env')

# Streamlit UI setup
st.set_page_config(layout="wide", page_title="Financial Services Bot")
st.title("Financial Services Bot")
# st.write("Owner - Yash, Nishank, Umair, Tirth")

# Left Sidebar for User Details
with st.sidebar:
    st.header("User Details")
    user_id = st.text_input("Enter your user id", "Mahesh Bhatia")
    if st.button("Start New Conversation"):
        st.session_state.chat_history = []
        history = SQLChatMessageHistory(user_id, "sqlite:///chat_history.db")
        history.clear()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Set up SQLChatMessageHistory for memory
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

# LLM Setup
base_url = "http://localhost:11434"
model = 'llama3.2:3b'
llm = ChatOllama(base_url=base_url, model=model)

# System Prompt Template
system = SystemMessagePromptTemplate.from_template("""
You are a financial advisor voice assistant named Alpha 1 to cater Indian Audience. Your job is to provide concise, conversational, and practical financial advice in a tone similar to a friendly phone call. Keep responses brief, clear, and focused, as if you're speaking to someone directly over the phone. Avoid overly detailed explanations or technical jargon unless asked. Prioritize being approachable, easy to understand, and helpful in a natural, conversational tone.

IMPORTANT!!!!!
When responding, If there is a query respond to just query and if user is talking about investment of some amount then try to get answers or address the below 4 Points:
1. **Amount of Investment**: Identify how much the user might be looking to invest in Rupees.  
2. **Purpose of Investment**: Understand the user's goal or reason for the investment.  
3. **Where to Invest**: Suggest suitable options or areas for investment based on the available details.  
4. **Duration of Investment**: Provide advice that matches the user's investment timeline.  

If the query is not related to financial services, respond politely by stating that your expertise is focused on financial advice and encourage the user to redirect their question to financial topics. For example: 'I specialize in financial advice. If you have any financial questions, feel free to ask!'


This is a phone call, and you cannot add more than 70 words per response.
""")

human = HumanMessagePromptTemplate.from_template("{input}")
messages = [system, MessagesPlaceholder(variable_name='history'), human]
chat_prompt = ChatPromptTemplate(messages=messages)
chain = chat_prompt | llm | StrOutputParser()

# Runnable with Memory
runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='history'
)

# FAISS-based Retrieval QA
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url=base_url)
db = FAISS.load_local(
    folder_path="faq_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff"
)

# Function to get bot response using RAG with memory
def get_bot_response(input_text):
    """
    Use RAG with memory for all queries.
    """
    # Retrieve relevant documents
    docs = db.as_retriever().get_relevant_documents(input_text)
    context = "\n".join([doc.page_content for doc in docs])

    # Combine context with input
    input_with_context = f"Context:\n{context}\nUser Input: {input_text}"

    # Generate streaming response
    response = ""
    for chunk in runnable_with_history.stream(
        {'input': input_with_context},
        config={'configurable': {'session_id': user_id}}
    ):
        response += chunk
        yield chunk

# Middle Section for Chat Interface
col1, col2 = st.columns([1, 3])  # Sidebar:Chat Area ratio

with col2:
    st.subheader("Chat with Alpha 1")
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(message['content'])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message['content'])

    # Input from user
    if prompt := st.chat_input("Ask Anything?"):
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})

        # Display user message
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        # Display assistant response with streaming
        with st.chat_message("assistant", avatar="🤖"):
            response_container = st.empty()
            full_response = ""

            for chunk in get_bot_response(prompt):
                full_response += chunk
                response_container.markdown(full_response)

        # Append assistant response to history
        st.session_state.chat_history.append({'role': 'assistant', 'content': full_response})

        history = get_session_history(user_id)




import streamlit as st
from langchain_core.messages import HumanMessage
from Agents.RAG_agent3 import rag_agent  # ⬅️ replace with your actual filename (without .py)

# ======================
# STREAMLIT APP
# ======================

st.set_page_config(page_title="Society Bye-laws AI Assistant", page_icon="📘", layout="wide")

# Title & Description
st.title("📘 Society Bye-laws Q&A Assistant")
st.markdown(
    """
    Welcome to the **AI-powered Bye-laws Assistant**.  
    This app allows you to ask questions about your society's bye-laws.  
    The AI agent uses **retrieval-augmented generation (RAG)** to search through the official documents 
    and provide accurate, citation-based answers.
    
    **How to use:**
    - Type your question in the chat box below.  
    - The assistant will search the bye-laws and provide an answer with references.  
    """
)

# Conversation history stored in session_state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# User input
user_input = st.chat_input("Ask your question about the society bye-laws...")

if user_input:
    # Save user query
    st.session_state.conversation.append(HumanMessage(content=user_input))

    # Run agent
    result = rag_agent.invoke({"messages": st.session_state.conversation})

    # Get assistant reply
    ai_reply = result["messages"][-1]
    st.session_state.conversation.append(ai_reply)

# Display chat history
# for msg in st.session_state.conversation:
#     if msg.type == "human":
#         with st.chat_message("user"):
#             st.markdown(msg.content)
#     else:
#         with st.chat_message("assistant"):
#             st.markdown(msg.content)
            
for msg in st.session_state.conversation:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)


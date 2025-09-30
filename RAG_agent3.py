from dotenv  import load_dotenv
import os
from langgraph.graph  import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph.message   import add_messages
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(model ="gpt-4o-mini", temperature = 0) #to get deterministic output so that model does not hallucinate - temperature = 0

#Embedding model has to be compatible with LLM
embeddings = OpenAIEmbeddings( model = "text-embedding-3-small")

import streamlit as st

#pdf_path = ["MODEL_BYE_LAWS.pdf", "MCS_Amendment_2019.pdf"]
pdf_folder = os.getenv("pdf_folder")

#get all pdf files from the folder
pdf_files = [os.path.join(pdf_folder,f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

#check collected files
pdf_names = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
print("pdf files:", pdf_names)


all_docs = []

for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs = loader.load() #returns a lsit of documents
    all_docs.extend(docs) #to to master list


try:
     #pages =  all_docs.load()
     print(f"Loaded {len(all_docs)} pages from {len(pdf_files)} Pdfs.")
except Exception as e:
     print(f"Error loading PDF: {e}")
     raise


#Chunking Process
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

pages_split = text_splitter.split_documents(all_docs)
persist_directory = os.getenv("persist_directory")
collection_name = "society_bye-laws"


#If our collection does not exist in the directory, we create using the os command
   
    
# === Vectorstore ===
vectorstore = None
try:
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        # Load existing Chroma DB
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        print("✅ Loaded existing ChromaDB vector store!")
    else:
        # Create a new Chroma DB and persist it
        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print("✅ Created new ChromaDB vector store!")

except Exception as e:
    print(f"❌ Error setting up ChromaDB: {str(e)}")

if vectorstore is None:
    raise RuntimeError("Vectorstore could not be created. Check Chroma setup.")
        
    
    #now we create our retriever
retriever = vectorstore.as_retriever(search_type = "similarity",search_kwargs = {"k": 8})#k is the amount of chunks to return


@tool
def retriever_tool(query:str) -> str:
    """
    this tool searches and returns information from the Society Bye-laws document
    """
    
    docs = retriever.invoke(query)
    
    if not docs:
        return "i   found no relevant information in the society bye-laws document"
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: \n{doc.page_content}")
    
    return "\n\n".join(results) 

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
def should_continue(state: AgentState) -> AgentState:
    """check if the last message contains tool  calls"""
    result  = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls)>0


system_prompt = """
You are an intelligent AI assistant who answers questions about  society's bye-laws based on the pdf document loaded
into your knowledge base. Use the retriver tool available to answer questions about the  society bye-laws related information. You can make
multiple calls if needed.  If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers 
"""


tools_dict = {our_tool.name: our_tool for our_tool in tools} #creating dictionary of our tools


#LLM Agent

def call_llm(state: AgentState)-> AgentState:
    """function to call the LLM with the current state"""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages':[message]}
    
    
    
#Retriever Agent

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from LLM's response"""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f'Calling tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}')
        
        if not t['name'] in tools_dict:  #check if valid tool is present
            print(f'\nTool: {t['name']} does not exist')
            result = "Incorrect tool name, please retry and select tool  from list of available tools"
            
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f' Result length: {len(str(result))}')
            
            
        #appends the tool message
        results.append(ToolMessage(tool_call_id= t['id'], name = t['name'], content = str(result)))
        
    print("Tools  execution complet, bacck to model")
    return {'messages': results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# def running_agent():
#     print("\n ===== RAG AGENT =====")
    
#     while True:
#         user_input = input("\nWhat is your question: ")
#         if user_input.lower() in ['exit','quit']:
#             break
        
#         messages = [HumanMessage(content = user_input)] #converts back to human message type
#         result = rag_agent.invoke({"messages": messages})
#         print("\n=== ANSWER ===")
#         print(result['messages'][-1].content)
        
# running_agent()
                             
                             
def running_agent():
    print("\n ===== RAG AGENT with Memory =====")

    # Store conversation history
    conversation = []

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit','quit']:
            break

        # Add user input to conversation history
        conversation.append(HumanMessage(content=user_input))

        # Run agent with the whole history
        result = rag_agent.invoke({"messages": conversation})

        # Get assistant reply
        ai_reply = result['messages'][-1]
        print("\n=== ANSWER ===")
        print(ai_reply.content)

        # Save assistant message into history
        conversation.append(ai_reply)


if __name__== "__main__":
    running_agent()
                                   
            
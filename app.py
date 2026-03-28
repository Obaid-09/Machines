import streamlit as st 
import sys
import os
from typing import TypedDict, List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# 1. MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="AI Study Assistant", layout="wide")

# (Optional) Use Streamlit Secrets or Environment Variables instead of hardcoding
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# 2. CACHE THE EXPENSIVE RAG SETUP
@st.cache_resource
def setup_rag():
    with open("rag.txt", "r", encoding="utf-8") as f:
        text = f.read()

    documents = [Document(page_content=text)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    chunks = []
    for doc in documents:
        split_chunks = text_splitter.split_documents([doc])
        chunks.extend(split_chunks)

    for i, chunk in enumerate(chunks):
        chunk.metadata["page"] = str(i + 1)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    # Upgrade to MMR to prevent duplicate/overlapping concepts like "slip" vs "slip ring"
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 20}
    )
    
    return retriever

retriever = setup_rag()

# 3. LLM SETUP
llm = ChatGroq(model="llama-3.1-8b-instant")

# 4. LANGGRAPH STATE & NODES
class AgentState(TypedDict):
    question: str
    rewritten_question: str 
    documents: List[Document]
    answer: str
    needs_retrieval: bool
    chat_history: List[Tuple[str, str]]
    mode: str


def retrieve_documents(state: AgentState) -> AgentState:
    question = state["rewritten_question"]
    documents = retriever.invoke(question)
    return {**state, "documents": documents}


def rewrite_question(state: AgentState) -> AgentState:
    question = state["question"]
    history = state.get("chat_history", [])

    if not history:
        return {**state, "rewritten_question": question}

    history_text = "\n".join([f"Q: {q} A: {a}" for q, a in history[-2:]])
    prompt = f"Rewrite the question so it is fully self-contained.\nConversation:\n{history_text}\nCurrent Question:\n{question}\nRewritten Question:\n"
    
    response = llm.invoke(prompt)
    return {**state, "rewritten_question": response.content.strip()}


def rewrite_question(state: AgentState) -> AgentState:
    question = state["question"]
    history = state.get("chat_history", [])

    # If no history, skip rewriting
    if not history:
        return {**state, "rewritten_question": question}

    # FIX: Only feed the previous *questions* to the rewriter, NOT the giant answers.
    # This stops the LLM from getting distracted by engineering terms in the answer.
    history_context = ""
    for q, a in history[-2:]:
        history_context += f"Previous Topic: {q}\n"

    prompt = f"""
You are an expert AI assistant. Your ONLY job is to rewrite the user's current question to make it fully self-contained by resolving pronouns (like "it", "this", "they").

Previous Questions asked by User:
{history_context}

Current Question:
{question}

Strict Rules:
1. Replace words like "it" with the actual subject from the Previous Questions.
2. DO NOT answer the question.
3. DO NOT add new topics or guess what the user wants to know beyond the exact text.
4. If the Current Question is already self-contained (no pronouns), just output the Current Question exactly as it is.

Rewritten Question:
"""
    response = llm.invoke(prompt)
    rewritten = response.content.strip()
    
    # Strip any markdown or weird quotes the LLM might add
    rewritten = rewritten.replace("**", "").replace('"', '')

    return {**state, "rewritten_question": rewritten}


def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    documents = state.get("documents", [])
    mode = state.get("mode", "normal")
    
    if documents:
        context = "\n\n".join([doc.page_content for doc in documents[:5]]) # Using your larger context
        history = state.get("chat_history", [])
        history_text = ""

        if history:
            history_text = "\n\nPrevious Conversation:\n"
            for q, a in history[-2:]: # Only show last 2 to save context window
                history_text += f"User: {q}\n"

        if mode in ["5mark", "5marks", "5 mark", "5 marks"]:
            mode_instruction = "Answer in a 5-mark exam format with moderate detail."
        elif mode in ["10mark", "10marks", "10 mark", "10 marks"]:
            mode_instruction = "Answer in a 10-mark exam format with detailed explanation."
        elif mode == "revision":
            mode_instruction = "Give a short and quick revision summary."
        else:
            mode_instruction = "Provide a standard, direct answer."

        prompt = f"""
You are a strict academic grading assistant. Your ONLY source of truth is the provided Context. 

Context:
{context}

Question:
{question}

Instruction: 
{mode_instruction}

STRICT RULES:
1. ZERO HALLUCINATION: You must not invent, guess, or pull from external knowledge. DO NOT create examples, do not invent RPM values, and do not invent calculations. 
2. MISSING DATA: If a specific section (like Formula) is not mentioned in the Context, you MUST write "N/A". 
3. NO MARKDOWN: You are strictly forbidden from using markdown. No bolding, no italics, no bullet points.
4. ABORT CONDITION: If the Context does not contain the answer, output ONLY: "The answer is not available in the provided material." Do not output anything else.

Format your answer EXACTLY as follows in plain text:

Definition: [Insert Definition or N/A]
Formula: [Insert Formula or N/A]
Explanation: [Insert Explanation or N/A]
Key Points: [Insert Key Points as plain text separated by commas]
"""
    else:
        prompt = f"Answer the following question in plain text without markdown: {question}"
    
    response = llm.invoke(prompt)
    answer = response.content
    
    # Clean up the weird edge case where it appends the abort text to a good answer
    if "Definition:" in answer and "The answer is not available" in answer:
        answer = answer.replace("The answer is not available in the provided material.", "").strip()
    
    pages = sorted({doc.metadata.get("page", "Unknown") for doc in documents})
    answer += "\n\n-------------\nSource Pages: " + ", ".join(pages)
    
    chat_history = state.get("chat_history", [])
    chat_history.append((question, answer))

    return {**state, "answer": answer, "chat_history": chat_history}

workflow = StateGraph(AgentState)
workflow.add_node("rewrite", rewrite_question)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# 5. FIXED ASK QUESTION FUNCTION (Now accepts 3 arguments)
def ask_question(question: str, history: List, mode: str):
    initial_state = {
        "question": question,
        "chat_history": history,
        "mode": mode,
        "documents": [],
        "answer": "",
        "needs_retrieval": False,
    }
    
    final_state = app.invoke(initial_state)
    return final_state["answer"], final_state["chat_history"]

# ==========================================
# 6. STREAMLIT UI
# ==========================================

st.title("😎 HELLO BUDDY, HOW CAN I HELP YOU")
st.sidebar.title("Settings")

subject = st.sidebar.selectbox("Select Subject", ["Machines"])
mode = st.sidebar.selectbox("Answer Mode", ["normal", "5mark", "10mark", "revision"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []

# Display Chat History
for role, message in st.session_state.chat_history:
    # Handle tuple format (Question, Answer) from LangGraph state
    if type(role) == str and type(message) == str:
        with st.chat_message("user"):
            st.write(role)
        with st.chat_message("assistant"):
            if "Source Pages" in message:
                parts = message.split("-------------")
                st.write(parts[0])
                if len(parts) > 1:
                    st.caption(parts[1])
            else:
                st.write(message)

# Handle Input
query = st.chat_input("Ask your question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking..."):
        answer, updated_history = ask_question(
            query,
            st.session_state.chat_history,
            mode
        )

    st.session_state.chat_history = updated_history

    with st.chat_message("assistant"):
        if "Source Pages" in answer:
            parts = answer.split("-------------")
            st.write(parts[0])
            if len(parts) > 1:
                st.caption(parts[1])
        else:
            st.write(answer)
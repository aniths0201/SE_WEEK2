import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
# Load the environment variables from the .env file
load_dotenv()

# Retrieve the API key using os.environ
api_key = os.environ.get("GEMINI_API_KEY")

# If API key not found throw error
if not api_key:
    raise ValueError("Error: Gemini API key not found. Check your .env file.")
    st.stop()


# 1. Page Configuration & UI Styling
st.set_page_config(page_title="EduPath Pro - Course Finder", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size: 2.6rem; color: #1E3A8A; font-weight: 700; text-align: center; margin-bottom: 5px; }
    .subtitle { font-size: 1.1rem; color: #4B5563; text-align: center; margin-bottom: 30px; }
    .guardrail-alert { background-color: #FEF2F2; padding: 15px; border-radius: 8px; border-left: 5px solid #DC2626; color: #991B1B; }
    </style>
""", unsafe_allow_html=True)

# 2. Hardcoded Knowledge Base (Curated Student Resources)
CURATED_KNOWLEDGE = [
    Document(page_content="Data Science & AI: Recommended platforms are Coursera (IBM Data Science Professional Certificate) and Kaggle Learn. For interactive Python, use DataCamp.", metadata={"source": "Data & AI Track"}),
    Document(page_content="Software Engineering & Web Development: Use freeCodeCamp for foundational HTML/CSS/JavaScript. Odin Project is highly rated for full-stack paths. Harvard's CS50 on edX is the gold standard for computer science basics.", metadata={"source": "Tech Track"}),
    Document(page_content="UI/UX & Graphic Design: Coursera Google UX Design Professional Certificate is top-rated. Use Figma's official YouTube learning channel and Interaction Design Foundation (IxDF).", metadata={"source": "Design Track"}),
    Document(page_content="Business & Digital Marketing: Google Digital Garage offers a free Fundamentals of Digital Marketing with certification. Coursera offers Wharton's Business Foundations specialization.", metadata={"source": "Business Track"}),
    Document(page_content="Product Management: Product School resources, Product Management Fundamentals on edX, and Reforge offer industry-standard training paths.", metadata={"source": "Product Track"})
]

# # 3. Setup Core AI, Embeddings, and Vector Store Function
@st.cache_resource
def init_llm_components():
    texts = ["Student resource document chunk 1", "Student resource document chunk 2"]
    
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    # 1. Build the FAISS store
    vector_store = FAISS.from_texts(texts=texts, embedding=embedding_model)
    
    # 2. Extract retriever
    retriever = vector_store.as_retriever()
    
    # 3. Seamlessly build the tool using the core function
    search_tool = create_retriever_tool(
        retriever,
        name="student_resource_search",
        description="Use this to search for answers within the student resources."
    )
    
    return llm, vector_store, search_tool
    
    # 5. Placeholder for your search tool definition
    search_tool = "your_search_tool_here" 
    
    return llm, vector_store, search_tool

    # 3. Create your metadatas if you are using them (must be the same length as texts, i.e., 2)
    metadatas = [{"source": "doc1"}, {"source": "doc2"}] 

    # Line 53: Now Python happily runs this because 'texts' exists!
    vector_store = FAISS.from_texts(
        texts=texts, 
        embedding=embedding_model, 
        metadatas=metadatas
    )
    return llm, vector_store, search_tool


# INITIALIZE COMPONENTS GLOBALLY BEFORE FUNCTIONS ARE CALLED ---
llm, vector_store, search_tool = init_llm_components()


# 4. Guardrail Logic Function
def evaluate_guardrails(user_input: str, llm_model) -> tuple[bool, str]:
    guardrail_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an academic safety guardrail system. Assess if the input is safe, ethical, and student-appropriate.
        Respond strictly in this JSON format:
        {{"safe": true/false, "reason": "brief reason explaining why if unsafe, otherwise empty"}}
        """),
        ("human", "Evaluate this career request: {career}")
    ])
    chain = guardrail_prompt | llm_model | StrOutputParser()
    try:
        response = chain.invoke({"career": user_input})
        if '"safe": false' in response or '"safe":false' in response:
            data = json.loads(response)
            return False, data.get("reason", "Unsuitable topic for a student platform.")
        return True, ""
    except:
        return True, ""


# 5. Hybrid Search & Synthesis Engine
def fetch_top_courses(career_goal: str, llm_model, v_store, s_tool) -> str:
    local_docs = v_store.similarity_search(career_goal, k=2)
    local_context = "\n".join([doc.page_content for doc in local_docs])
    
    search_query = f"top rated online courses for becoming a {career_goal} site:coursera.org OR site:udemy.com OR site:edx.org"
    web_context = s_tool.run(search_query)
    
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an encouraging student career guidance counselor. 
        Your task is to provide course recommendations based on the verified local data and live web search data below.
        
        Structuring Requirements:
        1. Welcome the student warmly.
        2. Highlight any pathways matching our Local Trusted Knowledge Base.
        3. Present 3-4 top course suggestions with clickable markdown link formatting (e.g., [Course Name](url)).
        4. Give a summary of what they'll learn.
        5. Keep the formatting exceptionally clean and readable using bullet points.
        """),
        ("human", "Desired Career: {career}\n\n[Local Knowledge Base Context]:\n{local_context}\n\n[Live Web Context]:\n{web_context}")
    ])
    
    chain = synthesis_prompt | llm_model | StrOutputParser()
    return chain.invoke({
        "career": career_goal, 
        "local_context": local_context, 
        "web_context": web_context
    })


# 6. Streamlit UI Presentation Layer
st.markdown("<div class='main-title'>🎓 EduPath Explorer Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Curated institutional roadmaps enhanced with real-time web verification</div>", unsafe_allow_html=True)

with st.form("search_form"):
    student_input = st.text_input(
        "What career or profession do you wish to pursue?", 
        placeholder="e.g., Data Scientist, UI/UX Designer, Software Engineer...",
    )
    submit_button = st.form_submit_button("Find My Path 🚀")

if submit_button:
    if not student_input.strip():
        st.warning("Please enter a valid career path first!")
    else:
        with st.spinner("Analyzing input safety via guardrails..."):
            is_safe, refusal_msg = evaluate_guardrails(student_input, llm)
            
        if not is_safe:
            st.markdown(f"<div class='guardrail-alert'><strong>🚫 Guardrail Block:</strong> {refusal_msg}</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Searching local knowledge base & verified web channels..."):
                try:
                    course_recommendations = fetch_top_courses(student_input, llm, vector_store, search_tool)
                    st.success("Your personalized roadmap is ready! ✨")
                    st.markdown(course_recommendations)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

st.markdown("---")
st.caption("Hybrid RAG Application Powered by FAISS Vector Storage, LangChain, and Gemini 2.5 Flash.")
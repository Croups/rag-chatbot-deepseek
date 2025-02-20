import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import time
import os
from datetime import datetime


from pathlib import Path
PDF_STORAGE_DIR = "document_store/pdfs"


def create_directories():
    Path(PDF_STORAGE_DIR).mkdir(parents=True, exist_ok=True)

def save_uploaded_file(uploaded_file):
    try:
        create_directories()
        file_path = os.path.join(PDF_STORAGE_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

# Page configuration
st.set_page_config(
    page_title="Reason your way through the document : RAG with DeepSeek",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Global styles */
    .stApp {
        background-color: white;
    }
    
    /* Sidebar styling */
    [data-testid=stSidebar] {
        background-color: #f8f9fa;
        padding: 2rem 1rem;
        border-right: 1px solid #e6e6e6;
    }
    
    /* Header styling */
    h1 {
        color: #0056b3;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    h2, h3 {
        color: #0056b3;
        margin-top: 1rem;
    }
    
    /* Thinking process container */
    .thinking-container {
        background-color: #f0f8ff;
        border-left: 4px solid #0056b3;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        white-space: pre-wrap;
        border-radius: 5px;
    }
    
    /* Final answer container */
    .final-answer {
        background-color: #ffffff;
        border: 2px solid #0056b3;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* Process header */
    .process-header {
        color: #0056b3;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    /* Status indicator */
    .status-indicator {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .status-active {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-inactive {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* Chat container */
    .chat-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #e6e6e6;
        height: 600px;
        overflow-y: auto;
    }
    
    /* Message styling */
    .stChatMessage {
        background-color: #f8f9fa !important;
        border: 1px solid #e6e6e6 !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    
    /* Input field styling */
    .stTextInput input {
        border: 2px solid #0056b3;
        border-radius: 5px;
        padding: 0.5rem;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #0056b3;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #004494;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #0056b3;
        margin: 1rem 0;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Error message styling */
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Metrics container */
    .metrics-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e6e6e6;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session states
if 'chatbot_initialized' not in st.session_state:
    st.session_state.chatbot_initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {
        'total_chunks': 0,
        'total_tokens': 0,
        'response_times': []
    }

# Prompt templates
THINKING_TEMPLATE = """
You are a logical reasoning assistant. Let's think about this step by step:

1. First, understand what's being asked
2. Review the relevant context
3. Break down the key points
4. Analyze the relationships
5. Form a reasoned conclusion

Context: {context}
Question: {question}

Let's reason through this:
"""

FINAL_ANSWER_TEMPLATE = """
Based on the above analysis, provide a clear and concise answer.
Focus on the most relevant points and ensure clarity.

Previous analysis: {thinking}
Question: {question}

Final answer:
"""

def stream_text(text, container_class):
    """Stream text with visual feedback"""
    placeholder = st.empty()
    displayed_text = ""
    
    for line in text.split('\n'):
        displayed_text += line + '\n'
        placeholder.markdown(
            f'<div class="{container_class}">{displayed_text}</div>',
            unsafe_allow_html=True
        )
        time.sleep(0.05)
    
    return displayed_text

# Sidebar configuration
with st.sidebar:
    st.title("🛠️ Setup")
    
    # API Key input with validation
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    if groq_api_key:
        if len(groq_api_key) < 20:
            st.error("Please enter a valid API key")
        else:
            os.environ["GROQ_API_KEY"] = groq_api_key
    
    # Model selection with information
    st.markdown("### 🤖 Model Selection")
    model_options = {
        "deepseek-r1-distill-qwen-32b": "Balanced performance", 
        "deepseek-r1-distill-llama-70b": "Best for complex tasks",
        "llama3-70b-8192": "Extended context window",
        "gemma2-9b-it": "Fast and efficient"
    }
    selected_model = st.selectbox(
        "Select Groq Model",
        options=list(model_options.keys()),
        help="Choose a model based on your needs"
    )
    st.info(f"💡 {model_options[selected_model]}")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                              help="Higher values make output more creative")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100,
                             help="Size of text chunks for processing")
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50,
                                help="Overlap between chunks")
    
    # File upload with progress
    st.markdown("### 📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type="pdf",
        help="Upload your PDF document for analysis"
    )
    
    if uploaded_file:
        st.markdown(f"📁 File: **{uploaded_file.name}**")
        st.markdown(f"📊 Size: **{uploaded_file.size/1024:.1f} KB**")
    
    if uploaded_file and groq_api_key and selected_model:
        initialize = st.button("🚀 Initialize Chatbot")
        if initialize:
            with st.spinner("Initializing chatbot..."):
                try:
                    start_time = time.time()
                    
                    # Save uploaded file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    if file_path:
                        # Load and process document
                        loader = PDFPlumberLoader(file_path)
                        documents = loader.load()
                        
                        # Split text
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        splits = text_splitter.split_documents(documents)
                        
                        # Create vector store
                        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                        vectorstore = InMemoryVectorStore(embeddings)
                        vectorstore.add_documents(splits)
                        
                        # Initialize Groq model
                        llm = ChatGroq(
                            model_name=selected_model,
                            temperature=temperature
                        )
                        
                        # Store components in session state
                        st.session_state.vectorstore = vectorstore
                        st.session_state.llm = llm
                        st.session_state.chatbot_initialized = True
                        
                        # Update processing stats
                        st.session_state.processing_stats['total_chunks'] = len(splits)
                        st.session_state.processing_stats['init_time'] = time.time() - start_time
                        
                        st.success("✅ Chatbot initialized successfully!")
                        st.info(f"📁 File saved to: {file_path}")
                        st.info(f"📊 Chunks created: {len(splits)}")
                    else:
                        st.error("Failed to save uploaded file")
                        
                except Exception as e:
                    st.error(f"Error during initialization: {str(e)}")


# Main chat interface
if st.session_state.chatbot_initialized:
    st.title("💬 RAG Chatbot")
    
    # System status and metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", st.session_state.processing_stats['total_chunks'])
    with col2:
        avg_response_time = sum(st.session_state.processing_stats['response_times'] or [0]) / (len(st.session_state.processing_stats['response_times']) or 1)
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    with col3:
        st.metric("Total Conversations", len(st.session_state.chat_history))
    
    # Chat container
    st.markdown("### 💭 Chat History")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    if "thinking" in message:
                        st.markdown('<p class="process-header">🤔 Thinking Process:</p>', unsafe_allow_html=True)
                        st.markdown(f'<div class="thinking-container">{message["thinking"]}</div>', unsafe_allow_html=True)
                        st.markdown('<p class="process-header">✨ Final Answer:</p>', unsafe_allow_html=True)
                        st.markdown(f'<div class="final-answer">{message["answer"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                start_time = time.time()
                
                # Search relevant documents
                docs = st.session_state.vectorstore.similarity_search(prompt)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Generate thinking process
                thinking_prompt = ChatPromptTemplate.from_template(THINKING_TEMPLATE)
                thinking_chain = thinking_prompt | st.session_state.llm
                thinking_response = thinking_chain.invoke({
                    "context": context,
                    "question": prompt
                })
                
                # Stream thinking process
                st.markdown('<p class="process-header">🤔 Thinking Process:</p>', unsafe_allow_html=True)
                thinking_text = stream_text(thinking_response.content, "thinking-container")
                
                # Generate final answer
                final_prompt = ChatPromptTemplate.from_template(FINAL_ANSWER_TEMPLATE)
                final_chain = final_prompt | st.session_state.llm
                final_response = final_chain.invoke({
                    "thinking": thinking_text,
                    "question": prompt
                })
                
                # Stream final answer
                st.markdown('<p class="process-header">✨ Final Answer:</p>', unsafe_allow_html=True)
                final_text = stream_text(final_response.content, "final-answer")
                
                # Update stats
                response_time = time.time() - start_time
                st.session_state.processing_stats['response_times'].append(response_time)
                
                # Store in chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "thinking": thinking_text,
                    "answer": final_text,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

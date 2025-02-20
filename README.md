# DeepSeek RAG Chatbot

A streamlined Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and LangChain, using DeepSeek models through Groq's API. This application allows users to upload PDF documents and interact with them through natural language queries, featuring step-by-step reasoning and streaming responses.

# DEMO 

https://github.com/user-attachments/assets/c91ed6f5-521d-462a-97b4-be821761dd93

## 🌟 Features

- **PDF Document Processing**: Upload and analyze PDF documents
- **Step-by-Step Reasoning**: Watch the AI think through each question
- **Streaming Responses**: Real-time response generation
- **Advanced Settings**: Customize model parameters and processing options
- **Performance Metrics**: Track response times and processing statistics
- **Clean UI**: Modern, responsive interface with blue and white theme

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key
- OpenAI API key (for embeddings)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Croups/rag-chatbot-deepseek
cd deepseek-rag-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
deepseek-rag-chatbot/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables
├── README.md             # Project documentation
└── document_store/       # Document storage directory
    └── pdfs/            # PDF storage directory
```

## 💡 Usage Guide

1. **Setup**
   - Enter your Groq API key
   - Select a DeepSeek model
   - Configure advanced settings (optional)

2. **Document Upload**
   - Upload a PDF document
   - Documents are stored in `document_store/pdfs/`
   - System processes and indexes the document

3. **Chatting**
   - Ask questions about your document
   - View the AI's thinking process
   - Get clear, concise answers

4. **Advanced Settings**
   - Temperature: Control response creativity (0.0-1.0)
   - Chunk Size: Adjust text processing (500-2000)
   - Chunk Overlap: Set context overlap (0-500)

## 🔧 Available Models

- **deepseek-r1-distill-qwen-32b**: Balanced performance
- **deepseek-r1-distill-llama-70b**: Best for complex tasks
- **llama3-70b-8192**: Extended context window
- **gemma2-9b-it**: Fast and efficient

## 🔍 Technical Details

### Components

- **Frontend**: Streamlit
- **RAG Implementation**: LangChain
- **Embeddings**: OpenAI Text Embeddings
- **LLM Provider**: Groq
- **PDF Processing**: PDFPlumber
- **Text Splitting**: RecursiveCharacterTextSplitter

### Process Flow

1. Document Upload → PDF Processing → Text Chunking
2. Chunk Embedding → Vector Storage
3. Query Processing → Context Retrieval
4. LLM Processing → Streaming Response

## ⚙️ Configuration Options

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Temperature | 0.0-1.0 | 0.7 | Controls response randomness |
| Chunk Size | 500-2000 | 1000 | Text chunk size for processing |
| Chunk Overlap | 0-500 | 200 | Overlap between chunks |

## 🛠️ Troubleshooting

Common issues and solutions:

1. **API Key Errors**
   - Verify API key length and format
   - Check environment variable configuration

2. **PDF Processing Issues**
   - Ensure PDF is not password protected
   - Check file permissions
   - Verify PDF is not corrupted

3. **Memory Issues**
   - Reduce chunk size for large documents
   - Process smaller sections if needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Support

Feel free to contact me on linkedin : www.linkedin.com/in/enes-koşar


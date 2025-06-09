# RAG Chatbot Prototype

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and chat with them using semantic search and AI-powered responses.

## Features

- 📄 **PDF Document Upload**: Upload and process multiple PDF documents
- 🔍 **Semantic Search**: FAISS-powered vector similarity search
- 🤖 **AI Chat Interface**: Natural language conversations with your documents
- 📚 **Source Citations**: See which documents informed each response
- 💾 **Persistent Storage**: Vector database persists between sessions
- ⚙️ **Configurable Settings**: Customize models, chunking, and retrieval parameters

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd rag-agent
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the application**:
   ```bash
   streamlit run main.py --server.address=0.0.0.0
   ```

The application will open in your browser at `http://localhost:8501`.

## Usage

### 1. Upload Documents
- Go to the **Documents** page
- Upload one or more PDF files
- Wait for processing and embedding generation

### 2. Chat with Documents
- Switch to the **Chat** page
- Ask questions about your uploaded documents
- View source citations for each response

### 3. Manage Knowledge Base
- View document statistics
- Remove unwanted documents
- Clear the entire knowledge base

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_COUNT=5
TEMPERATURE=0.7
MAX_TOKENS=1000
```

### Model Configuration

- **Embedding Model**: `text-embedding-ada-002` (default)
- **Chat Model**: `gpt-3.5-turbo` (default)
- **Vector Database**: FAISS with cosine similarity

## Project Structure

```
rag-agent/
├── main.py                     # Streamlit application entry point
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variable template
├── config/                    # Configuration files
│   ├── settings.py            # Application settings
│   └── logging.py             # Logging configuration
├── src/                       # Application source code
│   ├── services/              # Business logic services
│   │   ├── document_service.py # PDF processing
│   │   ├── embedding_service.py # OpenAI embeddings
│   │   ├── vector_service.py   # FAISS vector database
│   │   ├── rag_service.py      # RAG pipeline
│   │   └── chat_service.py     # Conversation management
│   ├── models/                # Data models
│   │   ├── document.py        # Document and chunk models
│   │   ├── conversation.py    # Chat message models
│   │   └── search.py          # Search result models
│   ├── utils/                 # Utility functions
│   │   ├── pdf_utils.py       # PDF processing utilities
│   │   ├── text_utils.py      # Text chunking utilities
│   │   └── openai_utils.py    # OpenAI API helpers
│   └── ui/                    # Streamlit UI components
│       ├── components/        # Reusable UI components
│       └── pages/             # Page definitions
└── data/                      # Local data storage
    ├── uploads/               # Temporary file storage
    ├── vector_db/             # FAISS index persistence
    └── logs/                  # Application logs
```

## Architecture

The application follows a modular architecture with clear separation of concerns:

- **Document Processing**: PDF text extraction and chunking
- **Embedding Generation**: OpenAI API integration for vector embeddings
- **Vector Storage**: FAISS database for efficient similarity search
- **RAG Pipeline**: Retrieval-augmented generation using LangChain patterns
- **Chat Management**: Conversation state and history management
- **UI Components**: Streamlit-based user interface

## Dependencies

### Core Dependencies
- `streamlit`: Web application framework
- `langchain`: RAG pipeline orchestration
- `faiss-cpu`: Vector similarity search
- `openai`: OpenAI API client
- `pypdf2`: PDF text extraction
- `pydantic`: Data validation and models

### Supporting Libraries
- `python-dotenv`: Environment variable management
- `numpy`: Numerical operations
- `tiktoken`: Token counting for OpenAI models

## Troubleshooting

### Common Issues

**No API Key Error**:
- Ensure `OPENAI_API_KEY` is set in your `.env` file
- Check that the API key starts with `sk-`

**Upload Failures**:
- Check file size (max 50MB)
- Ensure files are valid PDF format
- Check available disk space

**Slow Performance**:
- Reduce chunk size for faster processing
- Lower retrieval count for faster responses
- Process documents one at a time

**No Search Results**:
- Ensure documents are fully processed
- Check that embeddings were generated successfully
- Verify vector database contains vectors

### Performance Tips

- **For Large Documents**: Process one at a time to avoid memory issues
- **For Better Accuracy**: Use larger chunk sizes with more overlap
- **For Faster Responses**: Reduce retrieval count and chunk size
- **For Memory Efficiency**: Clear old documents periodically

## Development

### Running Tests
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/
```

### Adding New Features
1. Implement service logic in `src/services/`
2. Create data models in `src/models/`
3. Add UI components in `src/ui/components/`
4. Update main application in `main.py`

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Application
```bash
# Start the Streamlit application
streamlit run main.py --server.address=0.0.0.0

# Alternative port configuration
streamlit run main.py --server.port=8502
```

### Testing
```bash
# Run all tests
python -m pytest

# Run unit tests only
python -m pytest tests/unit/

# Run integration tests only
python -m pytest tests/integration/

# Run specific test file
python -m pytest tests/unit/test_document_service.py

# Run with verbose output
python -m pytest -v
```

### Code Quality
```bash
# Note: This project does not have linting/formatting configured
# Consider adding tools like black, flake8, or ruff for code quality
```

## Architecture

### Core Components
The application follows a **monolithic architecture** with clear service layer separation:

- **Services Layer** (`src/services/`): Core business logic
  - `document_service.py`: PDF processing and text extraction
  - `embedding_service.py`: OpenAI embedding generation
  - `vector_service.py`: FAISS vector database operations
  - `rag_service.py`: LangChain RAG pipeline orchestration
  - `chat_service.py`: Conversation management

- **Models Layer** (`src/models/`): Pydantic data models
  - `document.py`: Document and chunk models
  - `conversation.py`: Chat message and session models
  - `search.py`: Search result and context models

- **UI Layer** (`src/ui/`): Streamlit interface components
  - `components/`: Reusable UI components
  - `pages/`: Page definitions (chat, documents, settings)

### Key Design Patterns
- **Pipeline Pattern**: Document processing flows through extract → chunk → embed → store
- **Service Layer**: Business logic separated from UI and data access
- **Session State Management**: Streamlit session state maintains application state
- **Repository Pattern**: Vector database operations abstracted behind interfaces

### Data Flow
1. User uploads PDF via Streamlit UI
2. Document service extracts text and creates chunks
3. Embedding service generates vectors via OpenAI API
4. Vector service stores embeddings in FAISS index
5. Chat queries trigger semantic search and LLM response generation

## Configuration

### Environment Variables
Required in `.env` file:
- `OPENAI_API_KEY`: OpenAI API key for embeddings and chat
- `CHUNK_SIZE`: Text chunk size (default: 1000)
- `CHUNK_OVERLAP`: Chunk overlap size (default: 200)
- `RETRIEVAL_COUNT`: Number of chunks to retrieve (default: 5)
- `TEMPERATURE`: LLM temperature (default: 0.7)
- `MAX_TOKENS`: Max response tokens (default: 1000)

### Settings Management
Configuration is centralized in `config/settings.py` using Pydantic settings with automatic `.env` file loading.

## Key Dependencies
- **streamlit**: Web application framework
- **langchain**: RAG pipeline orchestration
- **faiss-cpu**: Vector similarity search
- **openai**: OpenAI API client
- **pypdf2**: PDF text extraction
- **pydantic**: Data validation and models

## Development Notes

### Session State Management
Streamlit session state is used extensively for:
- Service initialization and persistence
- Conversation history
- Document processing state
- Application configuration

### Error Handling
- Services implement graceful degradation when OpenAI API key is missing
- Document processing includes status tracking and error messaging
- UI displays appropriate error states and recovery options

### Testing Strategy
- Unit tests focus on service layer functionality
- Integration tests verify OpenAI API integration and RAG pipeline
- Test structure follows pytest conventions

### Local Data Storage
- `data/uploads/`: Temporary uploaded file storage
- `data/vector_db/`: FAISS index persistence
- `data/logs/`: Application logs

## Common Development Patterns

### Adding New Services
1. Create service class in `src/services/`
2. Define related models in `src/models/`
3. Add service initialization in `main.py`
4. Create corresponding UI components if needed

### Document Processing Extension
The document processing pipeline is designed for extensibility:
- New document types can be added by extending `DocumentService`
- Different chunking strategies can be implemented
- Alternative embedding models can be configured

### UI Component Development
- Follow existing component patterns in `src/ui/components/`
- Use Streamlit session state for component state management
- Implement error boundaries and loading states
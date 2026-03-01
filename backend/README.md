# EmpathEase Backend

Affective computing backend for emotional support and therapeutic interaction.

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- CUDA-capable GPU (recommended for ML inference)

### Setup

1. **Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start databases (Docker):**
```bash
docker-compose up -d
```

4. **Set up environment variables:**
```bash
copy .env.example .env
# Edit .env with your configuration
```

5. **Run database migrations:**
```bash
alembic upgrade head
```

6. **Start the server:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
backend/
├── app/                    # Main application
│   ├── api/               # API routes
│   ├── core/              # Core utilities
│   ├── models/            # SQLAlchemy models
│   ├── schemas/           # Pydantic schemas
│   ├── services/          # Business logic
│   ├── ml/                # ML inference
│   ├── memory/            # Memory systems
│   └── db/                # Database connections
├── training/              # ML training scripts
├── tests/                 # Test suite
└── docker-compose.yml     # Local services
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- WebSocket: ws://localhost:8000/ws/{session_id}

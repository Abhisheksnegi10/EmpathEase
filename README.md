# 🧠 EmpathEase

> **Real-time multimodal AI therapy assistant** — fusing facial affect, vocal prosody, and text emotion into a unified emotional intelligence engine.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)](https://vitejs.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ What is EmpathEase?

EmpathEase is an AI-powered mental wellness platform that understands how you feel — not just from what you *say*, but from how you *look* and *sound*. Three parallel emotion streams are continuously fused into a single coherent understanding of your emotional state, enabling a more empathetic, context-aware conversational experience.

| Stream | Technology | What it detects |
|---|---|---|
| 👁️ **Facial Affect** | ONNX · OpenCV | Valence, arousal, 7 core emotions |
| 🗣️ **Vocal Prosody** | LibROSA · PyTorch | Pitch, energy, tempo, emotional tone |
| 💬 **Text Emotion** | DistilBERT · GoEmotions | 28 fine-grained emotions from text |
| 🔀 **Fusion Engine** | Custom weighted fusion | Unified emotion state + suppression detection |

---

## 🏗️ Architecture

```
EmpathEase/
├── backend/                  # FastAPI + ML inference server
│   ├── app/
│   │   ├── api/routes/       # REST & WebSocket endpoints
│   │   ├── ml/               # Emotion inference modules
│   │   │   ├── facial_affect.py
│   │   │   ├── text_emotion.py
│   │   │   ├── vocal_prosody.py
│   │   │   ├── fusion.py
│   │   │   └── fear_disambiguator.py
│   │   ├── services/         # Business logic
│   │   ├── memory/           # Session & episodic memory
│   │   ├── schemas/          # Pydantic data models
│   │   └── core/             # Config, security, utils
│   ├── training/             # ML training scripts
│   └── tests/                # Integration & unit tests
└── frontend/                 # React + Vite UI
    ├── App.jsx               # Main application component
    └── src/
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Node.js | 18+ |
| Docker & Docker Compose | Latest |
| GPU (recommended) | CUDA-capable |

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/EmpathEase.git
cd EmpathEase
```

### 2. Backend setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your API keys (Groq, Sarvam) and DB credentials

# Start supporting services (PostgreSQL, Redis, Qdrant, Neo4j)
docker-compose up -d

# Run migrations
alembic upgrade head

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API Docs available at: **http://localhost:8000/docs**

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend available at: **http://localhost:5173**

---

## 🔑 Key Features

- **Real-time WebSocket streaming** — continuous emotion analysis via `ws://localhost:8000/ws/{session_id}`
- **Multimodal emotion fusion** — weighted combination of all three streams with confidence scoring
- **Fear disambiguation** — distinguishes between contextual fear (e.g. movies) and genuine distress
- **Suppression detection** — flags emotional masking when vocal/facial signals contradict text
- **Crisis detection** — configurable threshold-based safety triggers (`CRISIS_THRESHOLD=0.7`)
- **PII scrubbing** — automatic anonymisation of personally identifiable information via Presidio
- **Session memory** — Redis working memory + Qdrant episodic memory for context across sessions
- **LLM-powered responses** — Groq (Llama 3.3 70B) with Ollama fallback
- **Sarvam AI STT/TTS** — speech-to-text and text-to-speech in Indian languages

---

## 🌐 API Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/emotion/analyze` | Analyze text emotion |
| `POST` | `/api/v1/vocal/analyze` | Analyze vocal audio |
| `POST` | `/api/v1/privacy/scrub` | Scrub PII from text |
| `WS` | `/ws/{session_id}` | Real-time session stream |

Full interactive docs: **http://localhost:8000/docs** · **http://localhost:8000/redoc**

---

## ⚙️ Environment Variables

See [`backend/.env.example`](backend/.env.example) for the full list. Key variables:

```env
GROQ_API_KEY=          # LLM (required for chat)
SARVAM_API_KEY=        # STT/TTS (required for voice)
SECRET_KEY=            # App secret key
DATABASE_URL=          # PostgreSQL connection string
REDIS_URL=             # Redis connection string
CRISIS_THRESHOLD=0.7   # Safety sensitivity (0.0–1.0)
```

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## 🛠️ Tech Stack

**Backend:** FastAPI · PyTorch · ONNX Runtime · HuggingFace Transformers · LibROSA · OpenCV · LangChain · Groq · Sarvam AI · Presidio  
**Databases:** PostgreSQL · Redis · Qdrant · Neo4j  
**Frontend:** React 18 · Vite 5  
**Infrastructure:** Docker · Celery · Alembic

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Built for **AMD Slingshot 2026**
- Facial affect model trained on AffectNet / FER+ datasets
- Text emotion powered by [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) fine-tuned DistilBERT
- Speech processing via [Sarvam AI](https://sarvam.ai)

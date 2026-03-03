<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=00c9a7&height=180&section=header&text=EmpathEase&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Multimodal%20AI%20Mental%20Health%20Companion&descAlignY=58&descSize=18" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-amber?style=flat-square)]()

> **Real-time multimodal AI therapy assistant** that fuses facial affect, vocal prosody, and multilingual text emotion to deliver empathetic, context-aware mental wellness support — in Hindi, English, and Hinglish.

**Built solo. From scratch. End to end.**

</div>

---

## 📐 System Architecture

[**👉 View the full interactive architecture diagram here**](./empathease_architecture.html)

The system is structured across 6 production layers:

```
┌─────────────────────────────────────────────────┐
│  USER  →  Face Stream + Voice Input + Text       │
└──────────────────────┬──────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  LAYER 1 · PERCEPTION       │
        │  FER-2013 CNN (face)        │
        │  Sarvam STT (voice→text)    │
        │  MuRIL NLP (Hindi/Hinglish) │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  LAYER 2 · FUSION ENGINE    │
        │  Weighted emotional state   │
        │  Trajectory tracking        │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  LAYER 3 · POST-PROCESSING  │
        │  Cognitive distortion flags │
        │  Incongruence detection     │
        │  Needs-grounding flags      │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  LAYER 4 · CRISIS DETECTION │
        │  Hard-coded safety protocol │
        │  LLM bypassed at URGENT     │  ← deterministic, not AI
        │  iCall + Vandrevala lines   │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  LAYER 5 · THERAPY ENGINE   │
        │  Dev:  Llama 3.1 8B/Ollama  │
        │  Test: Groq API             │
        │  Prod: Claude API           │
        │  CBT + Person-Centered      │
        │  Hinglish code-switch aware │
        └──────────────┬──────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  LAYER 6 · INFRASTRUCTURE   │
        │  FastAPI · Docker · React   │
        │  WebSocket streaming        │
        │  PostgreSQL session storage │
        └─────────────────────────────┘
```

---

## ✨ Key Features

- 🎭 **Multimodal Emotion Recognition** — Fuses facial (CNN/FER-2013), vocal (Sarvam STT), and text (MuRIL) signals into a unified emotional state
- 🗣️ **Multilingual** — Hindi, English, Hinglish via MuRIL — 500M+ native speakers supported
- 🧠 **Emotional Memory** — Tracks emotional trajectory across a session for context-aware responses
- 🔀 **Incongruence Detection** — Surfaces gaps between face, voice, and text signals
- 🆘 **Crisis Safety Protocol** — Hard-coded at urgent level; LLM fully bypassed, helplines delivered instantly
- 🔒 **Privacy-First** — Local FER processing, biometric data never sold, DPDP Act 2023 compliant
- ⚡ **Low-Latency** — Locally hosted Llama 3 via Ollama, no external API calls in dev

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, TensorFlow, Keras, FER-2013 CNN |
| **NLP / LLMs** | MuRIL, Llama 3.1 8B, LangChain, Hugging Face Transformers |
| **Voice** | Sarvam STT (speech-to-text), Sarvam Bulbul TTS |
| **LLM Runtime** | Ollama (local dev), Groq API (test), Claude API (prod) |
| **Backend** | FastAPI, WebSocket, Python 3.10+, CUDA/CPU |
| **Frontend** | React, real-time FER overlay, MediaRecorder |
| **Infra** | Docker, PostgreSQL, Git |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Abhisheksnegi10/EmpathEase.git
cd EmpathEase

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model (local inference)
ollama pull llama3.1:8b

# Start the backend
uvicorn main:app --reload

# Start the frontend
cd frontend && npm install && npm start
```

---

## 🛡️ Ethics & Safety

| Principle | Implementation |
|-----------|---------------|
| **Crisis Protocol** | Hard-coded at urgent level — LLM bypassed, iCall + Vandrevala helpline delivered instantly |
| **Informed Consent** | Plain-language screen before every session |
| **Data Privacy** | Biometric data never stored externally; local FER; encrypted at rest + transit |
| **Not a Replacement** | Clear disclaimer every session — supportive tool, not a licensed therapist |
| **No Diagnosis** | Reflects emotional states only; never diagnoses DSM conditions |
| **Compliance** | DPDP Act 2023 (India's Digital Personal Data Protection Act) |

---

## 📊 Results

- ✅ **80%+ classification accuracy** on FER-2013 test set
- ✅ Multilingual emotion detection across Hindi, English, Hinglish
- ✅ Sub-second inference with locally-hosted Llama 3
- ✅ Emotional Memory module enabling context-aware multi-turn responses

---

## 👨‍💻 Author

**Abhishek Singh Negi**
Final year B.Tech AIML @ Sagar Institute of Research & Technology, Bhopal
Graduating June 2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/abhishek-singh-negi-733577304/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:abhisheksnegi10@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Abhisheksnegi10)

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=00c9a7&height=100&section=footer" width="100%"/>
</div>

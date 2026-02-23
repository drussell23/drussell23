<div align="center">

# Derek J. Russell

**AI Engineer | Computer Engineer | Autonomous Systems Architect**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/derek-j-russell/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/drussell23)
[![Featured](https://img.shields.io/badge/Mustang_News-Featured-8B0000?style=for-the-badge)](https://mustangnews.net/10-years-in-the-making-one-cal-poly-students-unique-path-to-an-engineering-degree/)

</div>

---

I design, build, and deploy production-grade AI ecosystems from bare metal to cloud inference. My work sits at the intersection of systems engineering, machine learning infrastructure, and real-time intelligent automation.

For the past 12 months, I have been executing a solo build of **JARVIS** — a three-repository, multi-process autonomous AI operating system spanning Python, C++, Rust, Swift, Objective-C, and TypeScript. The system orchestrates 60+ asynchronous agents across a neural mesh, routes inference dynamically between local Apple Silicon and GCP, performs real-time voice biometric authentication, controls macOS at the native API level, and continuously trains its own models through a self-improving feedback loop.

```
3,900+ commits  ·  2.5 million lines of code  ·  11 languages  ·  One engineer
```

---

## The JARVIS Ecosystem

JARVIS is not a chatbot wrapper. It is a distributed AI operating system composed of three interdependent repositories — each a standalone production system, together forming a self-improving autonomous intelligence.

### Architecture

```
                         UNIFIED SUPERVISOR KERNEL
                        (Single entry point, 50K+ LOC)
                                    |
              +---------------------+---------------------+
              |                     |                     |
         JARVIS (Body)       JARVIS-Prime (Mind)    ReactorCore (Forge)
         Python/Rust/Swift    Python / GGUF          C++ / Python
         Port 8010            Port 8000-8001         Port 8090
              |                     |                     |
     macOS native control    LLM inference routing    ML training engine
     60+ agent neural mesh   11 specialist models     LoRA / DPO / FSDP
     Voice biometrics        GCP golden image         GCP Spot VM recovery
     Ghost Display           Vision (LLaVA)           Deployment gate
     Computer use            Circuit breakers         Model lineage
     Google Workspace        Reasoning engine         Active learning
     Rust performance layer  Neural Switchboard       Causal discovery
     Autonomous agent loop   Hollow Client mode       Federated learning
```

### JARVIS — The Body

[`drussell23/JARVIS`](https://github.com/drussell23/JARVIS)

The central operating system and control plane. A custom Python kernel (`unified_supervisor.py`, 50K+ lines) that boots, coordinates, and monitors the entire ecosystem through a 7-zone initialization architecture with parallel dependency resolution.

<details>
<summary><b>Agent Architecture</b></summary>

- **Neural Mesh** — 16+ specialized agents (activity recognition, adaptive resource governor, context tracker, error analyzer, goal inference, Google Workspace, health monitor, memory, pattern recognition, predictive planning, spatial awareness, visual monitor, web search, coordinator) with asynchronous message passing, capability-based routing, and cross-agent data flow
- **Autonomous Agent Runtime** — multi-step goal decomposition, agentic task execution, tool orchestration, error recovery, and intervention decision engine with human-in-the-loop approval for destructive actions
- **AGI OS Coordinator** — proactive event stream, notification bridge, owner identity service, voice approval manager, and intelligent startup announcer

</details>

<details>
<summary><b>Voice and Authentication</b></summary>

- **Real-time voice biometric authentication** via ECAPA-TDNN speaker verification with cloud/local hybrid inference and multi-factor fusion (voice + proximity + behavioral)
- **Real-time voice conversation** — full-duplex audio (simultaneous mic + speaker), acoustic echo cancellation (speexdsp), streaming STT (faster-whisper), adaptive turn detection, barge-in control, and sliding 20-turn context window
- **Wake word detection** (Porcupine/Picovoice), Apple Watch Bluetooth proximity auth, continuous learning voice profiles
- **Unified speech state management** — STT hallucination guard, voice pipeline orchestration, parallel model loading

</details>

<details>
<summary><b>Vision and Spatial Intelligence</b></summary>

- **Never-skip screen capture** — two-phase monitoring (always-capture + conditional-analysis), self-hosted LLaVA multimodal analysis, Claude Vision escalation
- **Ghost Display** — virtual macOS display for non-intrusive background automation, Ghost Hands orchestrator for autonomous visual workflows
- **Claude Computer Use** — automated mouse, keyboard, and screenshot interaction via Anthropic's Computer Use API
- **OCR / OmniParser** — screen text extraction, window analysis, workspace name detection, multi-monitor and multi-space intelligence via yabai window manager
- **YOLO + Claude hybrid vision** — object detection with LLM-powered semantic understanding
- **Rust vision core** — native performance for fast image processing, bloom filter networks, and sliding window analysis

</details>

<details>
<summary><b>macOS Native Integration (Swift / Objective-C / Rust)</b></summary>

- **Swift bridge** (203 files) — CommandClassifier, SystemControl (preferences, security, clipboard, filesystem), PerformanceCore, ScreenCapture, WeatherKit, CoreLocation GPS
- **Objective-C voice unlock daemon** — JARVISVoiceAuthenticator, JARVISVoiceMonitor, permission manager, launchd service integration
- **Rust performance layer** — PyO3 bindings for memory pool management, quantized ML inference, vision fast processor, command classifier, health predictor; ARM64 SIMD assembly optimizations
- **CoreML acceleration** — on-device intent classification, voice processing

</details>

<details>
<summary><b>Infrastructure and Reliability</b></summary>

- **Parallel initializer** with cooperative cancellation, adaptive EMA-based deadlines, dependency propagation, and atomic state persistence
- **CPU-pressure-aware cloud shifting** — automatic workload offload to GCP when local resources are constrained
- **Enterprise hardening** — dependency injection container, enterprise process manager, system hardening, governance, Cloud SQL with race-condition-proof proxy management, TLS-safe connection factories, distributed lock manager
- **Three-tier inference routing**: GCP Golden Image (primary) → Local Apple Silicon (fallback) → Claude API (emergency)
- **Trinity event bus** — cross-repo IPC hub, heartbeat publishing, knowledge graph, state management, process coordination
- **Cost tracking and rate limiting** — GCP cost optimization with Bayesian confidence fusion, intelligent rate orchestration
- **File integrity guardian** — pre-commit integrity verification across the codebase

</details>

<details>
<summary><b>Intelligence and Learning</b></summary>

- **Google Workspace Agent** — Gmail read/search/draft, Google Calendar, natural language intent routing via tiered command router
- **Proactive intelligence** — predictive suggestions, proactive vision monitoring, proactive communication, emotional intelligence module
- **RAG pipeline** — ChromaDB vector store, FAISS similarity search, embedding service, long-term memory system
- **Chain-of-thought / reasoning graph engine** — LangGraph-based multi-step reasoning with conditional routing and reflection loops
- **Ouroboros** — self-programming engine for autonomous codebase analysis and improvement
- **Web research service** — autonomous web search and information synthesis
- **A/B testing framework** — vision pipeline experimentation
- **Repository intelligence** — code ownership analysis, dependency analyzer, API contract analyzer, AST transformer, cross-repo refactoring engine

</details>

### JARVIS-Prime — The Cognitive Core

[`drussell23/JARVIS-Prime`](https://github.com/drussell23/JARVIS-Prime)

The inference engine and reasoning layer. A production-ready model serving system that dynamically routes queries to specialist models based on task-type classification.

- **11 specialist GGUF models** (~40.4 GB) pre-baked into a GCP golden image with ~30-second cold starts
- **Task-type routing** — math queries hit Qwen2.5-7B, code queries hit DeepCoder, simple queries hit a 2.2 GB fast model, vision hits LLaVA
- **GCP Model Swap Coordinator** with intelligent hot-swapping, per-model configuration, and inference validation
- **Neural Switchboard v98.1** — stable public API facade over routing and orchestration with WebSocket integration contracts
- **Hollow Client mode** for memory-constrained hardware — strict lazy imports, zero ML dependencies at startup on 16 GB machines
- **Continuous learning hook** — post-inference experience recording for Elastic Weight Consolidation via ReactorCore
- **Reasoning engine activation** — chain-of-thought scaffolding (CoT/ToT/self-reflection) for high-complexity requests above configurable thresholds
- **APARS protocol** (Adaptive Progress-Aware Readiness System) — 6-phase startup with real-time health reporting to the supervisor
- **LLaVA vision server** — multimodal inference on port 8001 with OpenAI-compatible API, semaphore serialization, queue depth cap
- **Telemetry capture** — structured JSONL interaction logging with deployment feedback loop and post-deployment probation monitoring

### ReactorCore — The Forge

[`drussell23/JARVIS-Reactor`](https://github.com/drussell23/JARVIS-Reactor)

The ML training backbone. A hybrid C++/Python engine that transforms raw telemetry into improved models through an automated training pipeline with deployment safety gates.

- **Full training pipeline**: telemetry ingestion → active learning selection → gatekeeper evaluation → LoRA SFT → GGUF export → deployment gate → probation monitoring → feedback loop
- **DeploymentGate** validates model integrity before deployment; rejects corrupt or degenerate outputs
- **Post-deployment probation** — 30-minute health monitoring window with automatic commit or rollback based on live inference quality
- **Model lineage tracking** — full provenance chain (hash, parent model, training method, evaluation scores, gate decision) in append-only JSONL
- **Tier-2/Tier-3 runtime orchestration** — curriculum learning, meta-learning (MAML), causal discovery with correlation-based fallback, world model training
- **GCP Spot VM auto-recovery** with training checkpoint persistence and 60% cost reduction over on-demand instances
- **Native C++ training kernels** via CMake/pybind11/cpp-httplib for performance-critical operations
- **Atomic experience snapshots** — buffer drain under async lock, JSONL with DataHash for dataset versioning
- **PrimeConnector** — WebSocket path rotation, health polling fallback, contract path discovery for cross-repo communication
- **Cross-repo integration** — Ghost Display state reader, cloud mode detection, Trinity Unified Loop Manager, pipeline event logger with correlation IDs

---

## Technical Footprint

| Metric | Value |
|--------|-------|
| **Total commits** | 3,900+ across 3 repositories |
| **Codebase** | ~2.5 million lines across 11 languages |
| **Build duration** | 12 months, solo |
| **Unified kernel** | 50,000+ lines in a single orchestration file |
| **Neural Mesh agents** | 16+ specialized agents with async message passing |
| **Models served** | 11 specialist GGUF models via task-type routing |
| **Inference tiers** | GCP Golden Image → Local Metal GPU → Claude API |
| **Training pipeline** | Automated: telemetry → active learning → gatekeeper → training → GGUF export → deployment gate → probation → feedback |
| **Voice auth** | Multi-factor: ECAPA-TDNN biometric + Apple Watch proximity + behavioral analysis |
| **Vision pipeline** | Never-skip capture, LLaVA self-hosted, Claude escalation, YOLO hybrid, OCR/OmniParser |
| **Swift components** | 203 files — system control, command classifier, screen capture, GPS, weather |
| **Rust crates** | 5 Cargo workspaces — memory pool, vision processor, ML inference, SIMD optimizations |
| **Terraform modules** | 7 modules (compute, network, security, storage, monitoring, budget, Spot templates) |
| **Dockerfiles** | 6 (backend, backend-slim, frontend, training, cloud, GCP inference) |
| **GitHub Actions** | 20+ workflows (CI/CD, CodeQL, e2e testing, deployment, database validation, file integrity) |
| **macOS integration** | Native Swift/ObjC daemons, yabai WM, Ghost Display, multi-space/multi-monitor, launchd services |
| **Cloud infrastructure** | GCP (Compute Engine, Cloud SQL, Cloud Run, Secret Manager, Monitoring), Spot VM auto-recovery |
| **Google Workspace** | Gmail read/search/draft, Calendar, natural language routing via tiered command router |

---

## Tech Stack

<div align="center">

### Languages
[![Languages](https://skillicons.dev/icons?i=py,cpp,rust,swift,ts,js,bash,html,css&theme=dark)](https://skillicons.dev)

### ML, Inference and Data
[![ML](https://skillicons.dev/icons?i=pytorch,tensorflow,opencv,sklearn,anaconda&theme=dark)](https://skillicons.dev)

### Infrastructure and Cloud
[![Infra](https://skillicons.dev/icons?i=gcp,docker,kubernetes,terraform,redis,postgres,sqlite,nginx,cmake,githubactions&theme=dark)](https://skillicons.dev)

### Backend and Frontend
[![Backend](https://skillicons.dev/icons?i=fastapi,react,nextjs,nodejs,vscode,git,github,linux,apple,md&theme=dark)](https://skillicons.dev)

</div>

<details>
<summary><b>Full Stack Inventory (text)</b></summary>

**Languages:** Python, C++, Rust, Swift, Objective-C, TypeScript, JavaScript, SQL, Shell/Bash, AppleScript, HTML/CSS, ARM64 Assembly

**ML / Inference:** PyTorch, Transformers, llama.cpp, llama-cpp-python, GGUF quantization, ONNX Runtime, CoreML Tools, SpeechBrain, scikit-learn, SentenceTransformers, HuggingFace Hub, safetensors, tiktoken, Numba (JIT), sympy, LangChain, YOLO

**Training (ReactorCore):** LoRA, DPO, RLHF, FSDP, MAML (meta-learning), curriculum learning, federated learning, causal reasoning, world model training, online learning, active learning, EWC (Elastic Weight Consolidation)

**Models / Vision:** LLaVA (multimodal), ECAPA-TDNN (speaker verification), Whisper (faster-whisper, openai-whisper), Porcupine/Picovoice (wake word), Piper TTS, OmniParser (OCR)

**LLM APIs:** Anthropic Claude API (chat, vision, computer use), OpenAI API (chat completions, embeddings)

**Rust Ecosystem:** PyO3, ndarray, rayon (parallelism), parking_lot, DashMap (concurrent maps), crossbeam (lock-free channels), serde, mimalloc (allocator), image crate, ARM64 SIMD

**Swift / macOS Native:** Swift Package Manager, CoreLocation, WeatherKit, AppKit, Foundation, Quartz/CoreGraphics, Accessibility API, AVFoundation (video capture), pyobjc, launchd (service daemons), osascript (AppleScript bridge), yabai (tiling WM)

**Vector / Data:** ChromaDB, FAISS, Redis, PostgreSQL (asyncpg, psycopg2), SQLite (aiosqlite), NetworkX, bloom filters

**Infrastructure:** GCP (Compute Engine, Cloud SQL, Cloud Run, Secret Manager, Monitoring), Docker, docker-compose, Terraform (modular: compute, network, security, storage, monitoring, budget), Kubernetes, systemd, CMake, pybind11, cpp-httplib

**CI/CD:** GitHub Actions (20+ workflows: CI/CD pipeline, CodeQL analysis, e2e testing, deployment, database validation, file integrity), git worktrees

**Backend:** FastAPI, uvicorn, uvloop, gRPC, Protobuf, asyncio, aiohttp, httpx, WebSocket, Cloud SQL Proxy, circuit breakers, exponential backoff, backpressure control, distributed locks, epoch fencing

**Observability:** OpenTelemetry (tracing + metrics + OTLP/gRPC export), structlog, psutil, Pydantic, JSONL telemetry pipeline

**Frontend:** React 19, Next.js, Framer Motion, Axios, WebSocket real-time streaming

**Audio / Vision:** OpenCV, sounddevice, PyAudio, webrtcvad (VAD), speexdsp (AEC), librosa, pyautogui, LLaVA multimodal, YOLO object detection

</details>

---

## GitHub Stats

<div align="center">

<img src="https://github-readme-stats.vercel.app/api?username=drussell23&show_icons=true&theme=tokyonight&hide_border=true&count_private=true&include_all_commits=true" alt="GitHub Stats" height="180"/>
<img src="https://github-readme-stats.vercel.app/api/top-langs/?username=drussell23&layout=compact&theme=tokyonight&hide_border=true&langs_count=10" alt="Top Languages" height="180"/>

<img src="https://github-readme-streak-stats.herokuapp.com/?user=drussell23&theme=tokyonight&hide_border=true" alt="GitHub Streak"/>

</div>

---

## Background

I graduated from Cal Poly San Luis Obispo with a B.S. in Computer Engineering after a [10-year non-traditional academic path](https://mustangnews.net/10-years-in-the-making-one-cal-poly-students-unique-path-to-an-engineering-degree/) that started in remedial algebra at community college. I retook courses, studied through the loss of family, and spent most of my twenties earning a degree that others finish in four years. The path was not conventional. The outcome was.

JARVIS is what happens when that level of persistence meets engineering capability. Twelve months of daily commits, architectural decisions at every layer of the stack, and a refusal to ship anything that is not production-grade.

---

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/derek-j-russell/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-000000?style=flat-square&logo=vercel)](https://github.com/drussell23/JARVIS-Portfolio)
[![Article](https://img.shields.io/badge/Mustang_News-Read_Feature-8B0000?style=flat-square)](https://mustangnews.net/10-years-in-the-making-one-cal-poly-students-unique-path-to-an-engineering-degree/)

</div>

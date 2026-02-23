# Derek J. Russell

**AI Engineer | Computer Engineer | Autonomous Systems Architect**

I design, build, and deploy production-grade AI ecosystems from bare metal to cloud inference. My work sits at the intersection of systems engineering, machine learning infrastructure, and real-time intelligent automation.

For the past 12 months, I have been executing a solo build of **JARVIS** — a three-repository, multi-process autonomous AI operating system spanning Python, C++, Rust, and TypeScript. The system orchestrates 60+ asynchronous agents, routes inference dynamically between local Apple Silicon and GCP, and continuously trains its own models through a self-improving feedback loop.

**3,900+ commits. 2.5 million lines of code. One engineer.**

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
         Python / Rust        Python / GGUF          C++ / Python
         Port 8010            Port 8000-8001         Port 8090
              |                     |                     |
     macOS integration     LLM inference routing    ML training engine
     60+ agent mesh        11 specialist models     LoRA / DPO / FSDP
     Voice biometrics      GCP golden image         GCP Spot VM recovery
     Ghost Display         Vision (LLaVA)           Deployment gate
     Computer use          Circuit breakers         Model lineage
```

### JARVIS — The Body

[`drussell23/JARVIS`](https://github.com/drussell23/JARVIS)

The central operating system and control plane. A custom Python kernel (`unified_supervisor.py`, 50K+ lines) that boots, coordinates, and monitors the entire ecosystem through a 7-zone initialization architecture.

- **60+ agent communication mesh** with asynchronous message passing, capability-based routing, and cross-agent data flow
- **Real-time voice biometric authentication** via ECAPA-TDNN speaker verification with cloud/local hybrid inference
- **Vision pipeline** with never-skip screen capture, self-hosted LLaVA multimodal analysis, and Ghost Display for non-intrusive background automation
- **Parallel initializer** with cooperative cancellation, adaptive EMA-based deadlines, and dependency propagation
- **CPU-pressure-aware cloud shifting** — automatic workload offload to GCP when local resources are constrained
- **Enterprise hardening** — Cloud SQL with race-condition-proof proxy management, TLS-safe connection factories, atomic state persistence
- **Three-tier inference routing**: GCP Golden Image (primary) → Local Apple Silicon (fallback) → Claude API (emergency)

### JARVIS-Prime — The Cognitive Core

[`drussell23/JARVIS-Prime`](https://github.com/drussell23/JARVIS-Prime)

The inference engine and reasoning layer. A production-ready model serving system that dynamically routes queries to specialist models based on task-type classification.

- **11 specialist GGUF models** (~40.4 GB) pre-baked into a GCP golden image with ~30-second cold starts
- **Task-type routing** — math queries hit Qwen2.5-7B, code queries hit DeepCoder, simple queries hit a 2.2 GB fast model, vision hits LLaVA
- **GCP Model Swap Coordinator** with intelligent hot-swapping, per-model configuration, and inference validation
- **Hollow Client mode** for memory-constrained hardware — strict lazy imports, zero ML dependencies at startup on 16 GB machines
- **Continuous learning hook** — post-inference experience recording for Elastic Weight Consolidation via ReactorCore
- **Reasoning engine activation** — chain-of-thought scaffolding (CoT/ToT/self-reflection) for high-complexity requests above configurable thresholds
- **APARS protocol** (Adaptive Progress-Aware Readiness System) — 6-phase startup with real-time health reporting to the supervisor

### ReactorCore — The Forge

[`drussell23/JARVIS-Reactor`](https://github.com/drussell23/JARVIS-Reactor)

The ML training backbone. A hybrid C++/Python engine that transforms raw telemetry into improved models through an automated training pipeline with deployment safety gates.

- **Full training pipeline**: telemetry ingestion → active learning selection → gatekeeper evaluation → LoRA SFT → GGUF export → deployment gate → probation monitoring
- **DeploymentGate** validates model integrity before deployment; rejects corrupt or degenerate outputs
- **Post-deployment probation** — 30-minute health monitoring window with automatic commit or rollback based on live inference quality
- **Model lineage tracking** — full provenance chain (hash, parent model, training method, evaluation scores, gate decision) in append-only JSONL
- **Tier-2/Tier-3 runtime orchestration** — curriculum learning, meta-learning, causal discovery with correlation-based fallback
- **GCP Spot VM auto-recovery** with training checkpoint persistence and 60% cost reduction over on-demand instances
- **Native C++ training kernels** via CMake/pybind11 for performance-critical operations

---

## Technical Footprint

| Metric | Value |
|--------|-------|
| **Total commits** | 3,900+ across 3 repositories |
| **Codebase** | ~2.5 million lines (Python, C++, TypeScript, Rust) |
| **Build duration** | 12 months, solo |
| **Unified kernel** | 50,000+ lines in a single orchestration file |
| **Agent mesh** | 60+ concurrent asynchronous agents |
| **Models served** | 11 specialist GGUF models via task-type routing |
| **Inference tiers** | GCP Golden Image → Local Metal GPU → Claude API |
| **Training pipeline** | Automated: telemetry → training → deployment → probation → feedback |
| **Voice auth** | ECAPA-TDNN biometric verification with hybrid cloud/local inference |
| **Cloud infrastructure** | GCP (Compute Engine, Cloud SQL, Cloud Run), Spot VM auto-recovery |

---

## Stack

**Languages:** Python, C++, TypeScript, Rust

**ML / Inference:** llama.cpp, llama-cpp-python, PyTorch, Transformers, GGUF quantization, LoRA, DPO, FSDP, LLaVA, ECAPA-TDNN, SentenceTransformers

**Infrastructure:** GCP (Compute Engine, Cloud SQL, Cloud Run), Docker, Terraform, systemd, CMake, pybind11

**Backend:** FastAPI, WebSocket, asyncio, asyncpg, Cloud SQL Proxy, circuit breakers, exponential backoff

**Frontend:** React, Next.js, WebSocket real-time streaming

**Audio / Vision:** sounddevice, Whisper (faster-whisper), Piper TTS, speexdsp AEC, pyautogui, LLaVA multimodal

---

## Background

I graduated from Cal Poly San Luis Obispo with a B.S. in Computer Engineering after a [10-year non-traditional academic path](https://mustangnews.net/10-years-in-the-making-one-cal-poly-students-unique-path-to-an-engineering-degree/) that started in remedial algebra at community college. I retook courses, studied through the loss of family, and spent most of my twenties earning a degree that others finish in four years. The path was not conventional. The outcome was.

JARVIS is what happens when that level of persistence meets engineering capability. Twelve months of daily commits, architectural decisions at every layer of the stack, and a refusal to ship anything that is not production-grade.

---

## Contact

**LinkedIn:** [in/derek-j-russell](https://www.linkedin.com/in/derek-j-russell/)

**GitHub:** [drussell23](https://github.com/drussell23)

**Featured:** [Mustang News — "10 years in the making"](https://mustangnews.net/10-years-in-the-making-one-cal-poly-students-unique-path-to-an-engineering-degree/)

# 🛡️ NIDS 2.0: Autonomous Network Intrusion Detection System
### Dual-Tier AI Architecture: Supervised LightGBM + Unsupervised Deep Autoencoder

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](#)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20MPS%2FCUDA-ee4c2c.svg?logo=pytorch&logoColor=white)](#)
[![LightGBM](https://img.shields.io/badge/LightGBM-GBDT%20Signature-green.svg)](#)
[![Rust 2021](https://img.shields.io/badge/Rust-Wire--Speed%20Sniffer-black.svg?logo=rust&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Serving-009688.svg?logo=fastapi&logoColor=white)](#)
[![libpcap](https://img.shields.io/badge/libpcap-Real%20Wire%20Capture-orange.svg)](#)

---

## 🎯 Executive Summary & Problem Statement

Traditional signature-based firewalls (AWS WAF, Snort) can only detect attacks they have seen before. When an adversary launches a **Zero-Day exploit**, signature matching fails completely.

**NIDS 2.0** solves this through a **Dual-Tier Hybrid AI Defense System**:
1. **Tier-1 (LightGBM GBDT):** Identifies known signatures (`DoS`, `PortScan`, `BruteForce`) in sub-millisecond latency.
2. **Tier-2 (Deep Autoencoder):** Trained only on BENIGN traffic. Novel Zero-Day threats fail reconstruction, triggering quarantine.
3. **Latent Prototypes (KMeans):** 8 centroids in 16-D space for real-time novelty scoring.
4. **Wire-Speed Data Plane (Rust + libpcap):** Microsecond packet capture from physical NIC without Python GIL overhead.

---

## 🏗️ System Architecture

```
Physical NIC (en0/eth0)
        |
        | [libpcap - Zero Copy]
        v
  Rust Wire Engine        <-- Parses Ethernet/IP/TCP headers
  (engine/src/main.rs)        Aggregates into 5-tuple flows
        |
        | [HTTP POST - JSON]
        v
  Python AI Backend       <-- Tier-1: LightGBM signature match
  (backend/main.py)           Tier-2: Autoencoder anomaly score
        |
        | [WebSocket]
        v
  SOC Dashboard           <-- Real-time alerts, MSE graph, threat table
  (frontend/index.html)
```

---

## 📊 Evaluation Status

The checked-in training script generates synthetic flows; it does not train or
evaluate on CIC-IDS2017 or captured traffic. Its test predictions and zero-day
quarantine count have also been manually altered to produce boundary errors.
Therefore the previously listed 99.45% accuracy and 99.44% quarantine rate are
synthetic demo outputs, not valid IDS performance metrics. Real-capture accuracy,
macro-F1, balanced accuracy, per-class precision/recall, ROC-AUC, and false
positive rate have not been re-established from the files currently in this repo.
Do not present those demo figures as product or industry validation.

The intended next evaluation is to train on capture days that are disjoint from
the held-out capture day, report per-class metrics and false-positive rate, and
keep threshold/model selection separate from the final test capture. CIC-IDS2017
is available from the [official UNB CIC dataset page](https://www.unb.ca/cic/datasets/ids-2017.html).

---

## 🗂️ Project Structure

```
.
├── backend/
│   ├── adapter.py          # Feature Engineering: 77 raw → 91 dimensions
│   ├── risk_engine.py      # Dual-Tier inference (LightGBM + PyTorch)
│   ├── main.py             # FastAPI server + WebSocket
│   └── test_backend.py     # 4-class automated diagnostic test
├── engine/
│   ├── Cargo.toml          # Rust deps: pcap 2.5, reqwest, tokio, serde
│   └── src/main.rs         # libpcap sniffer + 5-tuple flow aggregator
├── frontend/
│   ├── index.html          # Dark-mode SOC dashboard
│   ├── styles.css          # Glassmorphism cyber design
│   └── app.js              # WebSocket + Chart.js live streams
├── models/
│   ├── autoencoder.pt      # PyTorch Autoencoder weights (80 KB)
│   ├── lightgbm_model.pkl  # LightGBM booster (776 KB)
│   ├── scaler.pkl          # RobustScaler (benign distribution)
│   ├── feature_list.json   # 91-feature canonical schema
│   ├── metadata.json       # Model threshold and latent prototypes
│   └── evaluation_dashboard.png
├── notebooks/
│   └── 01_nids_model_training.ipynb  # Google Colab training notebook
├── evaluation/             # UNSW-NB15 evaluation experiments and reports
├── data/                   # Dataset files are not included
├── scripts/
│   └── run_traffic_agent.py # Legacy Python attack emulator
├── requirements.txt
└── .gitignore
```

---

## 🚀 Complete Setup Guide

### Prerequisites

| Tool | Version | Install |
|:---|:---|:---|
| Python | 3.11+ | [python.org](https://python.org) |
| Rust + Cargo | 1.75+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| libpcap | System | macOS: pre-installed · Linux: `sudo apt install libpcap-dev` |

---

### Step 1 — Clone & Python Environment

```bash
git clone https://github.com/vipuljain675-projects/CyberAI.git
cd CyberAI

# Create virtual environment
python3 -m venv .venv

# Activate (run this in every new terminal session)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 2 — Install Rust & Build Wire Engine

```bash
# Install Rust (one-time setup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build Rust engine
cd engine
cargo build
cd ..
```

---

### Step 3 — Verify AI Models

```bash
.venv/bin/python backend/test_backend.py
```

This runs the repository's diagnostic checks against the bundled synthetic demo
models. Their output is not evidence of performance on captured or benchmark
traffic.

---

### Step 4 — Run the Full System

Open **2 terminals** and run simultaneously:

#### Terminal 1 — Python AI Backend
```bash
cd CyberAI
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open browser: **`http://127.0.0.1:8000`** → SOC Dashboard  
API Docs: **`http://127.0.0.1:8000/docs`**

#### Terminal 2 — Rust Wire Engine

```bash
cd CyberAI/engine
source "$HOME/.cargo/env"
```

| Mode | Command | Description |
|:---|:---|:---|
| **Demo / Showcase** (recommended) | `cargo run` | Infinite benchmark loop — sends DoS, PortScan, ZeroDay & Benign flows. Press `Ctrl+C` to stop. |
| **Real Live Wi-Fi** | `sudo cargo run -- --live` | libpcap captures real packets from default NIC |
| **Real Wi-Fi (specific interface)** | `sudo cargo run -- --live --dev en0` | Capture from `en0` (macOS Wi-Fi). Use `eth0` on Linux servers. |
| **PCAP File Replay** | `cargo run -- --pcap /path/file.pcap` | Replay a Wireshark `.pcap` dump |

---

### Find Your Network Interface (macOS)

```bash
ifconfig | grep -E "^[a-z]" | cut -d: -f1
```

Common interfaces:
- `en0` — Wi-Fi (use this for live capture)
- `en1` — Ethernet
- `utun0` — VPN tunnel (avoid — no real user traffic)

---

### Legacy Python Attack Emulator

If Rust is unavailable, the Python script sends attacks directly to the backend:

```bash
source .venv/bin/activate
python scripts/run_traffic_agent.py
# Press Ctrl+C to stop
```

---

## 🔬 Manual API Testing (curl)

```bash
# Benign flow
curl -s -X POST http://127.0.0.1:8000/api/v1/simulate/flow \
  -H "Content-Type: application/json" \
  -d '{"attack_type": "BENIGN"}' | python3 -m json.tool

# DoS attack
curl -s -X POST http://127.0.0.1:8000/api/v1/simulate/flow \
  -H "Content-Type: application/json" \
  -d '{"attack_type": "DoS"}' | python3 -m json.tool

# Zero-Day threat
curl -s -X POST http://127.0.0.1:8000/api/v1/simulate/flow \
  -H "Content-Type: application/json" \
  -d '{"attack_type": "ZeroDay"}' | python3 -m json.tool
```

---

## 🧠 Scientific Deep Dive

### Feature Engineering (77 → 91 Features)
14 derived interaction metrics:
- **Asymmetry Ratio:** `Fwd_Pkts / (Bwd_Pkts + 1)` — detects one-sided flood probes
- **Flag Density:** `Σ(SYN+FIN+RST+PSH+ACK+URG) / Total_Pkts` — abnormal TCP handshakes
- **Timing Jitter:** `Std(IAT) / (Mean(IAT) + 1)` — bot vs human rhythm
- **Log-Rate:** `log(1 + Pkts/s)` — compresses DoS dynamic ranges

### Autoencoder (Tier-2)
Architecture: `91 → 64 → 32 → 16 → 32 → 64 → 91` with BatchNorm1d + LeakyReLU(0.2).
The bundled operating threshold is read from `models/metadata.json`; it comes
from synthetic demonstration training and has not been validated on captured
traffic.

### Latent Prototypes
8 KMeans centroids in 16-D bottleneck. Novelty distance:
d(z) = min_k ‖z − Centroid_k‖₂

---

## 🛡️ Autonomous Threat Response

| Threat | Detection | Action | Mechanism |
|:---|:---|:---:|:---|
| BENIGN | Dual-Tier Clear | `ALLOW` | No penalty — passes through |
| Known class | Tier-1 LightGBM | Backend verdict | Reported to the API and dashboard |
| Anomalous flow | Tier-2 Autoencoder | Backend verdict | Reported to the API and dashboard |

---

## ⚠️ Known Limitations

### Evaluation and dataset limits
The bundled training script creates synthetic data. The evaluation scripts under
`evaluation/` analyze UNSW-NB15 experiments, but these do not validate the
bundled models on real-time captured traffic. Dataset download instructions,
experiment reports, and model artifacts should be interpreted within their
stated evaluation scope.

### sudo Required for NIC Access
`libpcap` requires root privileges for promiscuous mode — same requirement as Wireshark and tcpdump.

---

## 📜 Project Info
- **Project:** Autonomous Network Intrusion Detection System (NIDS 2.0)
- **Showcase:** Advanced AI / Cybersecurity — Xebia
- **Standard:** CICIDS2017 International Benchmark
- **Hardware:** Apple Silicon MPS · NVIDIA CUDA · CPU Fallback

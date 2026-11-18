# Kadak

Kadak is a research and demonstration network intrusion detection system that combines a Python inference service, a Rust flow-capture engine, and a browser-based monitoring dashboard.

The inference pipeline uses a LightGBM classifier for known traffic classes and a PyTorch autoencoder for anomaly scoring. A Rust engine aggregates packets into flows and sends them to the backend. The repository includes demonstration models and synthetic traffic tooling; the reported behavior should not be treated as validated detection performance on operational networks.

## Components

- **Capture and flow aggregation:** Rust, `pcap`, and a five-tuple flow representation.
- **Inference API:** FastAPI with LightGBM classification and autoencoder anomaly scoring.
- **Dashboard:** Browser interface receiving live updates over WebSocket.
- **Evaluation:** Synthetic demonstration data and UNSW-NB15 experiments. Results are limited to the datasets and procedures documented in the repository.

## Architecture

```text
Network interface or PCAP
          |
          v
Rust capture engine (engine/)
  Packet parsing and flow aggregation
          | HTTP / JSON
          v
FastAPI inference service (backend/)
  LightGBM classification and anomaly scoring
          | WebSocket
          v
Dashboard (frontend/)
```

## Repository layout

```text
backend/       FastAPI service, feature adapter, inference, and diagnostics
engine/        Rust packet capture and flow aggregation
frontend/      Dashboard assets
models/        Bundled demonstration model artifacts and metadata
notebooks/     Model training notebook
evaluation/    Dataset evaluation experiments and reports
scripts/       Traffic generation utilities
requirements.txt
```

Dataset files are not included. Model artifacts and synthetic examples are provided for development and demonstration, not as evidence of production readiness.

## Requirements

- Python 3.11 or later
- Rust and Cargo (1.75 or later)
- `libpcap` development libraries for live capture (for example, `libpcap-dev` on Debian or Ubuntu)

## Setup

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/vipuljain675-projects/CyberAI.git
cd CyberAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build the Rust engine:

```bash
cd engine
cargo build
cd ..
```

## Run

Start the backend from the repository root:

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The dashboard is served at `http://127.0.0.1:8000`; API documentation is at `http://127.0.0.1:8000/docs`.

In a second terminal, run the Rust engine in its default demonstration mode:

```bash
cd engine
cargo run
```

The engine also supports live capture and PCAP replay:

```bash
# Live capture from the default interface
sudo cargo run -- --live

# Live capture from a selected interface
sudo cargo run -- --live --dev eth0

# Replay a packet capture
cargo run -- --pcap /path/to/capture.pcap
```

Live capture may require elevated privileges and a platform-supported `libpcap` installation. Use only on networks and traffic that you are authorized to monitor.

The legacy Python traffic generator can be run instead of the Rust engine:

```bash
source .venv/bin/activate
python scripts/run_traffic_agent.py
```

## Evaluation scope

The checked-in training script generates synthetic flows. Its demonstration outputs, including model scores and quarantine counts, are not validated IDS performance metrics. Real-capture accuracy, macro-F1, balanced accuracy, per-class precision and recall, ROC-AUC, and false-positive rate have not been established for the bundled models.

The `evaluation/` directory contains UNSW-NB15 experiments. These experiments do not establish performance on live captured traffic. For a defensible benchmark, separate training, model selection, and final test captures; report per-class metrics and false-positive rates on held-out data. The [official CIC-IDS2017 dataset page](https://www.unb.ca/cic/datasets/ids-2017.html) provides one source of benchmark data.

## Implementation notes

- The feature adapter derives interaction features from flow statistics; the bundled schema is stored in `models/feature_list.json`.
- The autoencoder operating threshold and latent prototype metadata are stored in `models/metadata.json`. The bundled threshold comes from synthetic demonstration training and has not been validated on captured traffic.
- The API reports inference results to the dashboard. The repository should not be interpreted as implementing an independently validated network enforcement or quarantine control.

## License

Kadak is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the license text.

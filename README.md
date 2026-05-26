# SYMBIOTIC PLANETARY INTELLIGENCE (SPI)

**Lead Architect & Sole Owner:** bsalinder@gmail.com

SPI is an ultra-advanced bio-digital synthesis project that converts planetary-scale biological signal streams into semantic embeddings using a symbiotic liquid neural network architecture.

## SPI Pipeline

- Raw biological signals are filtered with a low-pass Butterworth filter.
- Channels are normalized using median and interquartile range scaling.
- A liquid neural network integrates temporal dynamics with manual Euler updates.
- The latent state is projected into a semantic concept embedding using a transformer bridge.
- The Flask API exposes `/predict` for signal inference and text fallback requests.

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   source venv/bin/activate      # macOS / Linux
   ```

2. Install project dependencies:
   ```bash
   make install
   ```

## Running Locally

```bash
make run
```

Visit:

```text
http://127.0.0.1:5000
```

## API Usage

### Signal inference

POST `/predict` with `signals` as a list of numeric arrays:

```json
{
  "signals": [
    [0.1, 0.2, 0.3, 0.4],
    [0.05, 0.1, 0.15, 0.2],
    [0.2, 0.25, 0.3, 0.35],
    [0.1, 0.05, 0.0, -0.05]
  ]
}
```

### Text fallback

POST `/predict` with a `text` field:

```json
{
  "text": "I need help"
}
```

## Running Tests

```bash
make test
```

## Docker

Build the container:

```bash
make docker-build
```

Run the container:

```bash
make docker-run
```

## Monitoring

Once the service is running, scrape metrics at:

```text
http://127.0.0.1:5000/metrics
```

The instrumentation exposes request counts and request latency histograms for `/predict`.

## Kubernetes

Apply the provided manifests to your cluster (ensure `kubectl` is configured):

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Or trigger the GitHub Actions `deploy-k8s.yml` workflow and provide a base64-encoded `KUBE_CONFIG` repository secret.

## Files

- `app.py` — Flask API and request validation
- `model.py` — SPI signal preprocessing and inference pipeline
- `Dockerfile` — production-style container image
- `Makefile` — automation for installation, testing and containerization
- `tests/` — unit tests for SPI inference and the API
- `.github/workflows/ci.yml` — automated CI pipeline
- `.github/workflows/tag-publish.yml` — tag-triggered Docker publish
- `.github/workflows/deploy-k8s.yml` — optional k8s deploy workflow (requires `KUBECONFIG` secret)
- `k8s/` — Kubernetes manifests

## LNN (torchdiffeq) Usage

The project includes an alternative LNN implementation using `torchdiffeq` in `lnn.py`.
This offers an ODE-based integrator for liquid neural dynamics and maps the final
latent state to a semantic vector.

Example:

```python
import torch
from lnn import SPI_OmniKernelODE

# prepare a short synthetic biosignal stream
T = 10
input_dim = 8
latent_dim = 16
sentiment_dim = 4

kernel = SPI_OmniKernelODE(input_dim, latent_dim, sentiment_dim)
x = torch.randn(T, input_dim)
t = torch.arange(0, T, dtype=torch.float32)

# returns a sentiment vector of length `sentiment_dim`
sentiment_vector = kernel.process_bio_stream(x, t)
print('sentiment vector shape:', sentiment_vector.shape)
```

Requirements: `torchdiffeq` must be installed (included in `requirements.txt`).

## Developer Note

Lead Architect & Sole Owner: bsalinder@gmail.com

This repository represents the SYMBIOTIC PLANETARY INTELLIGENCE (SPI) project. If
you contributed to this project (for example, Gemma), please ensure proper
attribution in commits and the CONTRIBUTORS file.

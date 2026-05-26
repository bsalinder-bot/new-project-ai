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

## Files

- `app.py` — Flask API and request validation
- `model.py` — SPI signal preprocessing and inference pipeline
- `Dockerfile` — production-style container image
- `Makefile` — automation for installation, testing and containerization
- `tests/` — unit tests for SPI inference and the API
- `.github/workflows/ci.yml` — automated CI pipeline

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

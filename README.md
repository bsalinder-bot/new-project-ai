# Planetary Translator

**Maintainer:** bsalinder@gmail.com

A lightweight AI service that processes biological signal inputs, performs stable neural integration, and returns semantic concept embeddings via a Flask API.

## Planetary Translator Pipeline

1. Raw biological signals are filtered using a low-pass Butterworth filter.
2. Signals are robustly normalized using median and interquartile range scaling.
3. A liquid neural network cell integrates the signal over time with manual Euler steps.
4. The final latent state is mapped to a semantic output using a transformer bridge.
5. The Flask app exposes a `/predict` endpoint for inference.

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   source venv/bin/activate      # macOS / Linux
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Run the app

```bash
make run
```

Then open:

```text
http://127.0.0.1:5000
```

## Testing

```bash
make test
```


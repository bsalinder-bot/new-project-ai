import math
import re
from collections import Counter, defaultdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from scipy import signal


class BioSignalProcessor:
    """Handles noise reduction and normalization of multi-channel biological signals."""

    def __init__(self, sampling_rate: int = 1000):
        self.sampling_rate = sampling_rate

    def apply_lowpass_filter(self, data: np.ndarray, cutoff: float = 50.0) -> np.ndarray:
        nyq = 0.5 * self.sampling_rate
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(5, normal_cutoff, btype='low', analog=False)
        padlen = 3 * max(len(a), len(b))
        if len(data) <= padlen:
            return data.astype(np.float32)
        return signal.filtfilt(b, a, data)

    def robust_normalize(self, data: np.ndarray) -> np.ndarray:
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return np.zeros_like(data)
        return np.clip((data - median) / iqr, -1.0, 1.0)

    def prepare_for_kernel(self, raw_signals: List[np.ndarray]) -> torch.Tensor:
        processed = []
        for s in raw_signals:
            if not isinstance(s, np.ndarray):
                raise ValueError('Each signal channel must be a numpy array.')
            if s.ndim != 1:
                raise ValueError('Each signal channel must be a 1D array.')
            filtered = self.apply_lowpass_filter(s)
            normed = self.robust_normalize(filtered)
            processed.append(normed)

        combined = np.stack(processed, axis=-1)
        return torch.tensor(combined, dtype=torch.float32).unsqueeze(0)


class LiquidBioCell(nn.Module):
    """LNN cell that computes the continuous-time derivative for latent dynamics."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        self.weight_in = nn.Linear(input_dim, hidden_dim)
        self.weight_rec = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        innovation = torch.tanh(self.weight_in(x) + self.weight_rec(y))
        return (-y / self.tau) + innovation


class SPI_OmniKernel(nn.Module):
    """Manual Euler integrator for the liquid neural network kernel."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.cell = LiquidBioCell(input_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = x_seq.shape
        y = torch.zeros(batch_size, self.latent_dim, device=x_seq.device)

        for t in range(time_steps):
            x_t = x_seq[:, t, :]
            dy_dt = self.cell(y, x_t)
            y = y + 0.001 * dy_dt

        return y


class TransformerBridge(nn.Module):
    """Maps the final latent state to a semantic concept embedding."""

    def __init__(self, latent_dim: int, semantic_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(256, semantic_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(latent)
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        hidden = hidden.unsqueeze(1)
        encoded = self.transformer(hidden)
        return self.head(encoded[:, 0, :])


class SymbioticPlanetaryIntelligence:
    """Combines signal preprocessing, liquid network integration, and concept projection."""

    def __init__(self, input_dim: int = 4, latent_dim: int = 512, semantic_dim: int = 256):
        self.processor = BioSignalProcessor()
        self.kernel = SPI_OmniKernel(input_dim, latent_dim)
        self.bridge = TransformerBridge(latent_dim, semantic_dim)
        self._validate_model()

    def _validate_model(self) -> None:
        dummy_input = torch.randn(1, self.kernel.latent_dim)
        _ = self.bridge(dummy_input)

    def infer(self, raw_signals: List[List[float]]) -> dict:
        if not raw_signals or not isinstance(raw_signals, list):
            raise ValueError('Raw signals must be a non-empty list of channels.')

        channels = []
        time_length = None
        for channel in raw_signals:
            if not isinstance(channel, list) or not channel:
                raise ValueError('Each channel must be a non-empty list of numeric values.')
            channel_array = np.asarray(channel, dtype=np.float32)
            if channel_array.ndim != 1:
                raise ValueError('Each channel must be one-dimensional.')
            if time_length is None:
                time_length = channel_array.shape[0]
            elif channel_array.shape[0] != time_length:
                raise ValueError('All channels must have the same length.')
            channels.append(channel_array)

        x = self.processor.prepare_for_kernel(channels)
        latent = self.kernel(x)
        concept = self.bridge(latent).detach().cpu().numpy()
        return {
            'embedding': concept.reshape(-1).tolist(),
            'shape': list(concept.shape),
            'status': 'ok',
            'channels': len(channels),
            'time_steps': x.shape[1]
        }


class SimpleIntentModel:
    """Fallback text classifier for UI and demonstration purposes."""

    def __init__(self):
        self.labels = []
        self.label_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.vocab = set()
        self._train()

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        words = re.findall(r"\b[a-z0-9']+\b", text)
        return [word for word in words if len(word) > 1]

    def _train(self) -> None:
        examples = [
            ("hi", "greeting"),
            ("hello", "greeting"),
            ("good morning", "greeting"),
            ("goodbye", "farewell"),
            ("see you later", "farewell"),
            ("thanks", "gratitude"),
            ("thank you", "gratitude"),
            ("i need help", "support"),
            ("can you assist me", "support"),
            ("i have a question", "support"),
            ("show me prices", "sales"),
            ("what are your rates", "sales"),
            ("i want to buy", "sales"),
            ("i feel sad", "emotion"),
            ("i am happy", "emotion"),
            ("this is frustrating", "emotion"),
        ]

        for text, label in examples:
            self.labels.append(label)
            self.label_counts[label] += 1
            tokens = self._tokenize(text)
            for token in tokens:
                self.word_counts[label][token] += 1
                self.vocab.add(token)

        self.total_examples = sum(self.label_counts.values())
        self.vocab_size = len(self.vocab)

    def _score(self, tokens: List[str], label: str) -> float:
        log_prob = math.log(self.label_counts[label] / self.total_examples)
        token_count = sum(self.word_counts[label].values())
        for token in tokens:
            count = self.word_counts[label][token] + 1
            log_prob += math.log(count / (token_count + self.vocab_size))
        return log_prob

    def predict(self, text: str) -> dict:
        tokens = self._tokenize(text)
        if not tokens:
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'tokens': [],
                'message': 'Please enter a longer sentence.'
            }

        scores = {label: self._score(tokens, label) for label in self.label_counts}
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        exp_scores = {label: math.exp(score - best_score) for label, score in scores.items()}
        total = sum(exp_scores.values())
        confidence = exp_scores[best_label] / total if total > 0 else 0.0

        return {
            'intent': best_label,
            'confidence': round(confidence, 3),
            'tokens': tokens,
            'message': 'Predicted user intent from text input.'
        }

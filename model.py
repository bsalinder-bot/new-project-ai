import math
import re
from collections import Counter, defaultdict


class SimpleIntentModel:
    def __init__(self):
        self.labels = []
        self.label_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.vocab = set()
        self._train()

    def _tokenize(self, text):
        text = text.lower()
        words = re.findall(r"\b[a-z0-9']+\b", text)
        return [word for word in words if len(word) > 1]

    def _train(self):
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

    def _score(self, tokens, label):
        log_prob = math.log(self.label_counts[label] / self.total_examples)
        token_count = sum(self.word_counts[label].values())
        for token in tokens:
            count = self.word_counts[label][token] + 1
            log_prob += math.log(count / (token_count + self.vocab_size))
        return log_prob

    def predict(self, text):
        tokens = self._tokenize(text)
        if not tokens:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "tokens": [],
                "message": "Please enter a longer sentence."
            }

        scores = {label: self._score(tokens, label) for label in self.label_counts}
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        exp_scores = {label: math.exp(score - best_score) for label, score in scores.items()}
        total = sum(exp_scores.values())
        confidence = exp_scores[best_label] / total if total > 0 else 0.0

        return {
            "intent": best_label,
            "confidence": round(confidence, 3),
            "tokens": tokens,
            "message": "Predicted user intent from text input."
        }

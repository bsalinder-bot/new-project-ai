import argparse
import json
import sys
from model import SymbioticPlanetaryIntelligence, SimpleIntentModel


def main():
    parser = argparse.ArgumentParser(description='SPI CLI for local inference')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str, help='Text input for fallback intent model')
    group.add_argument('--signals', type=str, help='Path to JSON file with signals list')

    args = parser.parse_args()

    if args.text:
        model = SimpleIntentModel()
        out = model.predict(args.text)
        print(json.dumps(out, indent=2))
        return

    if args.signals:
        with open(args.signals, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        signals = payload.get('signals')
        if not signals:
            print(json.dumps({'error': 'signals field missing or empty'}))
            sys.exit(1)

        spi = SymbioticPlanetaryIntelligence()
        out = spi.infer(signals)
        print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

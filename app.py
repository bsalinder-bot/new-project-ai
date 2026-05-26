import os
import time
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.exceptions import BadRequest
from model import SymbioticPlanetaryIntelligence, SimpleIntentModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__, static_folder='static', template_folder='templates')
spi_model = SymbioticPlanetaryIntelligence()
text_model = SimpleIntentModel()

# Prometheus metrics
REQUEST_COUNT = Counter('spi_requests_total', 'Total SPI requests', ['endpoint', 'method', 'status'])
REQUEST_LATENCY = Histogram('spi_request_latency_seconds', 'SPI request latency', ['endpoint'])


@app.route('/metrics')
def metrics():
    data = generate_latest()
    return Response(data, mimetype=CONTENT_TYPE_LATEST)

@app.errorhandler(BadRequest)
def handle_bad_request(error):
    return jsonify({
        'error': 'Bad request',
        'message': error.description or 'Invalid request payload.'
    }), 400

@app.errorhandler(500)
def handle_server_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again later.'
    }), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        raise BadRequest('Request body must be JSON.')

    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest('Invalid JSON payload.')
    start = time.perf_counter()
    endpoint = '/predict'
    try:
        if 'signals' in payload:
            signals = payload['signals']
            if not isinstance(signals, list) or not signals:
                raise BadRequest("The 'signals' field must be a non-empty list of numeric arrays.")

            try:
                result = spi_model.infer(signals)
            except ValueError as exc:
                raise BadRequest(str(exc))

        else:
            text = payload.get('text')
            if text is None or not isinstance(text, str):
                raise BadRequest("The request must include either 'signals' or a 'text' string.")

            result = text_model.predict(text)

        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(endpoint=endpoint, method='POST', status='200').inc()
        return jsonify(result)
    except BadRequest as e:
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(endpoint=endpoint, method='POST', status='400').inc()
        raise
    except Exception:
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(endpoint=endpoint, method='POST', status='500').inc()
        raise

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

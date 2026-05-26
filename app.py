import os
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import BadRequest
from model import SymbioticPlanetaryIntelligence, SimpleIntentModel

app = Flask(__name__, static_folder='static', template_folder='templates')
spi_model = SymbioticPlanetaryIntelligence()
text_model = SimpleIntentModel()

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

    if 'signals' in payload:
        signals = payload['signals']
        if not isinstance(signals, list) or not signals:
            raise BadRequest("The 'signals' field must be a non-empty list of numeric arrays.")

        try:
            result = spi_model.infer(signals)
        except ValueError as exc:
            raise BadRequest(str(exc))
        return jsonify(result)

    text = payload.get('text')
    if text is None or not isinstance(text, str):
        raise BadRequest("The request must include either 'signals' or a 'text' string.")

    result = text_model.predict(text)
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

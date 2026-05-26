from flask import Flask, render_template, request, jsonify
from model import SimpleIntentModel

app = Flask(__name__)
model = SimpleIntentModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json() or {}
    text = payload.get('text', '')
    result = model.predict(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

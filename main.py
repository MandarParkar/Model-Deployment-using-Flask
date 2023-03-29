from flask import request, url_for, redirect, Flask, render_template
from pickle import load
import numpy as np

with open('./venv/model.pkl', 'rb') as f:
    pipeline = load(f)


def get_results(review):
    result = pipeline.predict([review])
    return result[0]


app = Flask(__name__)


@app.route('/', methods=['GET'])
def show_index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def predict_result():
    review = request.form['review']
    result = get_results(review)
    return render_template('prediction.html', review=review, result=result)


if __name__ == '__main__':
    app.run(debug=True)

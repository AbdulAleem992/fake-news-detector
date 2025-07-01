from flask import Flask, render_template, request
import pickle

# Load the trained model and vectorizer
with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)

        result = "Real News ✅" if prediction[0] == 1 else "Fake News ❌"
        return render_template('index.html', prediction=result, input_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)

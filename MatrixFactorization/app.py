from flask import Flask,jsonify,render_template,request
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return model


if __name__ == "__main__":
    app.run(debug=True)

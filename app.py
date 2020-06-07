from flask import Flask,render_template,request
import pickle
import numpy as np
import random

np.random.seed(0)
random.seed(0)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
       For rendering results on HTML GUI
    '''
    UserID = request.form["UserID"]
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(UserID))


if __name__ == "__main__":
    app.run(debug=True)

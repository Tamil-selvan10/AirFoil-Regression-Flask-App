import numpy as np
import pandas as pd
import  pickle
from flask import Flask,render_template,url_for,request

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
        #return 'Hello World'
        return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(f'data:{data}')
    input=[list(data.values())]
    output=model.predict(input)[0]
    return {'pressure_level':np.round(output,2)}

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x)for x in request.form.values()]
    input=[data]
    output=model.predict(input)[0]
    return render_template('home.html',prediction_text=f'Airfoil Pressure:{np.round(output,2)}')

    


if __name__=='__main__':
    app.run(debug=True)

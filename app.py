from flask import Flask,request, url_for, redirect, render_template
from waitress import serve
import pickle
import numpy as np
app = Flask(__name__, template_folder='template')
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def hello_world():
    return render_template("indexx.html")
@app.route('/predict',methods=['POST','GET'])
#\nProbability of mental illness is {}'.format(output)
def predict():
    int_features=[int(x) for x in request.form.values()] #int to float
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    print(prediction)
    output='{0:.{1}f}'.format(prediction[0][1], 3)
    if output>str(0.485):
        return render_template('indexx.html',pred='You require counselling.')
    else:
        return render_template('indexx.html',pred='You are doing great.')
if __name__ == '__main__':
    app.run(debug=True)
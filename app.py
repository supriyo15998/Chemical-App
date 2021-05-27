from flask import Flask,jsonify,request,render_template
from tensorflow import keras
import os
import numpy as np
from PIL import Image
import random
import cv2
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/api/predict", methods=['POST'])
def predict():
    model = keras.models.load_model('multi2.h5')
    # model = keras.models.load_model('bin.h5')
    img = Image.open(request.files['name'])
    
    img.save('assets/incomingfile.jpg')
    # npimg = np.array(img)
    cvimg = cv2.imread("assets/incomingfile.jpg")

    res=cv2.resize(cvimg ,dsize=(500,500), interpolation=cv2.INTER_CUBIC)
    npimg=np.array(res)

    sw=np.moveaxis(npimg,0,0)
    rr=np.expand_dims(sw,0)
    print(npimg.shape)
    print("printing done")
    prediction = model.predict(rr)
    predlist = list(prediction[0])
    index = predlist.index(max(predlist))
    labels = ["benzene","acetaminophen","acetysalicylic","adrenaline","ethane","ethene","ethylene","ibuprofen","isopentane","propylene","M-xykene (1,3 - dimethylbenzene)",\
          "o-xylene (1,2 - dimethylbenzene)","neopentane","phenylalanine","P-xylene (1,4 - dimethylbenzene)","Unknown or Bonds"]
    print()
    result = {"result" : "The chemical is " + labels[index]}
    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=True)
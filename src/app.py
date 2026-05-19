from flask import Flask
import tensorflow as tf
import numpy as np

app=Flask(__name__)

model=tf.keras.models.load_model(
    "saved_model/model.keras"
)


@app.route('/')

def home():

    return "Reuters Transformer API Running"


@app.route('/predict')

def predict():

    sample=np.random.randint(
        1,
        10000,
        size=(1,200)
    )

    prediction=model.predict(
        sample
    )

    pred=int(
        np.argmax(
            prediction
        )
    )

    return {

        "predicted_class":pred
    }


app.run(

host='0.0.0.0',

port=5000
)

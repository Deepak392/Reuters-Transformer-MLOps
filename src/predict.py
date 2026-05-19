import tensorflow as tf
import numpy as np

model=tf.keras.models.load_model(
    "../saved_model/model.keras"
)

sample=np.random.randint(
    1,
    10000,
    size=(1,200)
)

prediction=model.predict(
    sample
)

print(
    np.argmax(prediction)
)

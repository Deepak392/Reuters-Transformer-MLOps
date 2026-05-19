from model import build_model
from preprocessing import load_data

import mlflow
import mlflow.tensorflow

x_train,y_train,x_test,y_test=load_data()

model=build_model()

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']
)

mlflow.set_experiment(
    "ReutersTransformer"
)

with mlflow.start_run():

    mlflow.log_param(
        "epochs",
        10
    )

    mlflow.log_param(
        "batch_size",
        64
    )

    history=model.fit(

        x_train,

        y_train,

        epochs=10,

        batch_size=64,

        validation_split=.2
    )

    acc=history.history[
        'accuracy'
    ][-1]

    loss=history.history[
        'loss'
    ][-1]

    mlflow.log_metric(
        "accuracy",
        acc
    )

    mlflow.log_metric(
        "loss",
        loss
    )

    model.save(
        "../saved_model/model.keras"
    )

    mlflow.tensorflow.log_model(
        model,
        "model"
    )

print("Training complete")

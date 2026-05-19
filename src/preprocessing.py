from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size=10000
maxlen=200

def load_data():

    (x_train,y_train),(
        x_test,y_test
    )=reuters.load_data(
        num_words=vocab_size
    )

    x_train=pad_sequences(
        x_train,
        maxlen=maxlen
    )

    x_test=pad_sequences(
        x_test,
        maxlen=maxlen
    )

    return x_train,y_train,x_test,y_test

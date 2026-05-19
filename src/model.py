import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

vocab_size=10000
maxlen=200

embed_dim=32
num_heads=4
ff_dim=64
num_classes=46


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        rate=0.1
    ):

        super().__init__()

        self.att=MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn=tf.keras.Sequential([

            Dense(
                ff_dim,
                activation="relu"
            ),

            Dense(embed_dim)
        ])

        self.norm1=LayerNormalization(
            epsilon=1e-6
        )

        self.norm2=LayerNormalization(
            epsilon=1e-6
        )

        self.drop1=Dropout(rate)

        self.drop2=Dropout(rate)

    def call(
        self,
        inputs,
        training=False
    ):

        attn=self.att(
            inputs,
            inputs
        )

        attn=self.drop1(
            attn,
            training=training
        )

        out1=self.norm1(
            inputs+attn
        )

        ffn=self.ffn(out1)

        ffn=self.drop2(
            ffn,
            training=training
        )

        return self.norm2(
            out1+ffn
        )


def build_model():

    inputs=Input(
        shape=(maxlen,)
    )

    x=Embedding(
        vocab_size,
        embed_dim
    )(inputs)

    x=TransformerEncoder(
        embed_dim,
        num_heads,
        ff_dim
    )(x)

    x=GlobalAveragePooling1D()(x)

    outputs=Dense(
        num_classes,
        activation='softmax'
    )(x)

    model=Model(
        inputs,
        outputs
    )

    return model

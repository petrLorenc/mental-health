import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LSTM, Concatenate, Lambda

from metrics import Metrics

module_url = "../resources/embeddings/use-4"


def build_lstm_model(hyperparams, hyperparams_features):

    # embedding_dim = hyperparams_features['embedding_dim']
    embeddings_dim = 512

    n_sentences = hyperparams['max_posts_per_user']
    # n_sentences = 4

    embedding_layer = hub.KerasLayer(module_url, trainable=True)

    input = tf.keras.layers.Input(shape=(n_sentences,), dtype=tf.string)
    x = [embedding_layer(input[:, s]) for s in range(n_sentences)]
    x = Concatenate(axis=1)(x)
    x = tf.keras.layers.Reshape((n_sentences, embeddings_dim))(x)
    x = LSTM(hyperparams['lstm_units_user'], return_sequences=False)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    metrics_class = Metrics(threshold=hyperparams['threshold'])
    model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                               metrics=[metrics_class.precision_m, metrics_class.recall_m,
                                        metrics_class.f1_m, AUC()])
    model.summary()
    return model


if __name__ == '__main__':
    n_sentences = 4
    model = build_lstm_model(hyperparams={"maxlen": n_sentences, "lstm_units_user": 64}, hyperparams_features=None)
    sentences = [str(i) for i in range(n_sentences)]
    X = [sentences, sentences[::-1]]  # 1 sample
    print(model.predict(X))
    # print(model.predict([[["a"], ["b"], ["c"]]]))

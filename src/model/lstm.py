import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LSTM, Concatenate, Lambda

from metrics import Metrics

hyperparams = {
    "trainable_embeddings": True,
    "dense_bow_units": 20,
    "dense_numerical_units": 20,
    "dense_user_units": 0,
    "dropout": 0.1,
    "l2_dense": 0.00011,
    "l2_embeddings": 1e-07,
    "norm_momentum": 0.1,
    "ignore_layer": [],

    "epochs": 50,
    "embeddings": "use",
    "positive_class_weight": 2,
    "maxlen": 50,
    "lstm_units_user": 100,
    "max_posts_per_user": 15,
    "batch_size": 64,

    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 55,
    "scheduled_reduce_lr_freq": 95,
    "scheduled_reduce_lr_factor": 0.5,
    "threshold": 0.5,

    "optimizer": "adam",
    "decay": 0.001,
    "lr": 5e-05,

    "padding": "pre"
}
hyperparams_features = {
    "module_url": "../resources/embeddings/use-4"
}


def build_lstm_model(hyperparams, hyperparams_features):
    n_sentences = hyperparams['max_posts_per_user']

    embedding_layer = hub.KerasLayer(hyperparams_features["module_url"], trainable=hyperparams["trainable_embeddings"])

    input = tf.keras.layers.Input(shape=(n_sentences,), dtype=tf.string)
    x = [embedding_layer(input[:, s]) for s in range(n_sentences)]
    x = Concatenate(axis=1)(x)
    x = tf.keras.layers.Reshape((n_sentences, hyperparams_features['embedding_dim']))(x)
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

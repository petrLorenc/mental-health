import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LSTM, Concatenate, Lambda

from metrics import Metrics

hyperparams = {
    "trainable_embeddings": False,
    "dropout": 0.1,
    "l2_dense": 0.00011,
    "l2_embeddings": 1e-07,
    "norm_momentum": 0.1,
    "ignore_layer": [],

    "epochs": 50,
    "embeddings": "use-raw",
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


def build_pure_lstm_model(hyperparams, hyperparams_features):
    n_sentences = hyperparams['max_posts_per_user']

    _input = tf.keras.layers.Input(shape=(n_sentences, hyperparams_features['embedding_dim'],))
    x = LSTM(hyperparams['lstm_units_user'], return_sequences=False)(_input)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=_input, outputs=_output)
    metrics_class = Metrics(threshold=hyperparams['threshold'])
    model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                  metrics=[metrics_class.precision_m, metrics_class.recall_m,
                           metrics_class.f1_m, AUC()])
    model.summary()
    return model

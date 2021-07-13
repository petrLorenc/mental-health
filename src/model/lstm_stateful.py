import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LSTM, Concatenate, Lambda, GRU

from metrics import Metrics

hyperparams = {
    "trainable_embeddings": False,
    "dropout": 0.1,
    "l2_dense": 0.00011,
    "l2_embeddings": 1e-07,
    "norm_momentum": 0.1,
    "ignore_layer": [],

    "epochs": 10,
    "embeddings": "use-stateful",
    "positive_class_weight": 2,
    "lstm_units_user": 100,
    "max_posts_per_user": 15,
    "batch_size": 1,

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


def build_lstm_stateful_model(hyperparams, hyperparams_features):
    _input = tf.keras.layers.Input(shape=(1, hyperparams_features['embedding_dim'],), batch_size=hyperparams["batch_size"])
    x = GRU(512, stateful=True)(_input)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=_input, outputs=_output)
    metrics_class = Metrics(threshold=hyperparams['threshold'])
    model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                  metrics=[metrics_class.precision_m, metrics_class.recall_m,
                           metrics_class.f1_m, AUC()])
    model.summary()
    return model

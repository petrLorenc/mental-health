import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC

from metrics import Metrics

hyperparams = {
    "positive_class_weight": 2,
    "max_posts_per_user": 15,
    "batch_size": 64,
    "epochs": 50,
    "embeddings": "bow",

    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 55,
    "scheduled_reduce_lr_freq": 95,
    "scheduled_reduce_lr_factor": 0.5,
    "threshold": 0.5,

    "optimizer": "adam",
}
hyperparams_features = {
    "vocabulary_path": "../resources/generated/vocab_20000_erisk.txt",
    "nrc_lexicon_path": "../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "liwc_path": "../resources/liwc.dic",
}


def build_bow_log_regression_model(hyperparams, hyperparams_features):

    _input = tf.keras.layers.Input(shape=(hyperparams_features["embedding_dim"],))
    # x = tf.keras.layers.Dense(64, activation="relu")(_input)
    # x = tf.keras.layers.Dense(32, activation="relu")(x)
    # _output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(_input)

    model = tf.keras.Model(inputs=_input, outputs=_output)
    metrics_class = Metrics(threshold=hyperparams['threshold'])
    model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                               metrics=[metrics_class.precision_m, metrics_class.recall_m,
                                        metrics_class.f1_m, AUC()])
    model.summary()
    return model


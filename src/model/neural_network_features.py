import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC

from metrics import Metrics

hyperparams = {
    "positive_class_weight": 2,
    "batch_size": 4,
    "epochs": 50,
    "embeddings": "unigrams-features",
    "dense_units": 128,

    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 55,
    "scheduled_reduce_lr_freq": 95,
    "scheduled_reduce_lr_factor": 0.5,
    "threshold": 0.5,

    "optimizer": "adam",
}
hyperparams_features = {
    "nrc_lexicon_path": "../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "liwc_path": "../resources/liwc.dic",
}


def build_neural_network_model_features(hyperparams, hyperparams_features, emotions_dim, liwc_categories_dim):

    _input_emotions = tf.keras.layers.Input(shape=(emotions_dim, ))
    _input_liwc = tf.keras.layers.Input(shape=(liwc_categories_dim, ))
    _input = tf.keras.layers.Input(shape=(hyperparams_features["embedding_dim"],))

    x_1 = tf.keras.layers.Dense(hyperparams['dense_units'], activation="relu")(_input)
    x_2 = tf.keras.layers.Dense(int(emotions_dim/2), activation="relu")(_input_emotions)
    x_3 = tf.keras.layers.Dense(int(liwc_categories_dim/2), activation="relu")(_input_liwc)
    x = tf.keras.layers.Concatenate()([x_1, x_2, x_3])

    _output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    # _output = tf.keras.layers.Dense(1, activation="sigmoid")(_input)

    model = tf.keras.Model(inputs=[_input, _input_emotions, _input_liwc], outputs=_output)
    metrics_class = Metrics(threshold=hyperparams['threshold'])
    model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                               metrics=[metrics_class.precision_m, metrics_class.recall_m,
                                        metrics_class.f1_m, AUC()])
    model.summary()
    return model


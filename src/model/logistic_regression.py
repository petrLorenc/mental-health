import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC

from train_utils.metrics import Metrics
from utils.default_config import DefaultHyperparameters

hyperparams = DefaultHyperparameters({
    "embeddings": "unigrams",
    "learning_rate": 0.001,
    "chunk_size": 100
})

hyperparams_features = {
    "nrc_lexicon_path": "../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "liwc_path": "../resources/liwc.dic",
}


def build_logistic_regression_model(hyperparams, hyperparams_features):

    _input = tf.keras.layers.Input(shape=(hyperparams_features["embedding_dim"],))
    # x = tf.keras.layers.Dense(64, activation="relu")(_input)
    # x = tf.keras.layers.Dense(32, activation="relu")(x)
    # _output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(_input)

    model = tf.keras.Model(inputs=_input, outputs=_output)
    metrics_class = Metrics(threshold=hyperparams['threshold'])
    model.compile(tf.optimizers.Adam(learning_rate=hyperparams["learning_rate"]) if "learning_rate" in hyperparams else hyperparams["optimizer"], K.binary_crossentropy,
                               metrics=[metrics_class.precision_m, metrics_class.recall_m,
                                        metrics_class.f1_m, AUC()])
    model.summary()
    return model


def log_important_features(experiment, vocabulary, model):
    weights = model.weights[0].numpy()
    important_words_weights = [(vocabulary[x], weights[x][0]) for x in np.argsort([(x[0]) for x in weights])[::-1]]
    for t, w in important_words_weights[:100]:
        experiment.log_text(t, metadata={"weight": str(w), "sign": "positive"})

    for t, w in important_words_weights[-100:]:
        experiment.log_text(t, metadata={"weight": str(w), "sign": "negative"})

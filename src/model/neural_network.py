import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC

from train_utils.metrics import Metrics
from utils.default_config import DefaultHyperparameters

hyperparams = DefaultHyperparameters({
    "embeddings": "",
    "dense_units": 32,
    "optimizer": "adam",
    "learning_rate": 0.01
})
hyperparams_features = {
    "nrc_lexicon_path": "../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "liwc_path": "../resources/liwc.dic",
}


def build_neural_network_model(hyperparams, hyperparams_features):

    _input = tf.keras.layers.Input(shape=(hyperparams_features["embedding_dim"],))
    x = tf.keras.layers.Dense(hyperparams['dense_units'], activation="relu")(_input)
    x = tf.keras.layers.Dropout(hyperparams['dropout_rate'])(x)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    # _output = tf.keras.layers.Dense(1, activation="sigmoid")(_input)

    model = tf.keras.Model(inputs=_input, outputs=_output)
    metrics_class = Metrics(threshold=hyperparams['threshold'])
    model.compile(tf.optimizers.Adam(learning_rate=hyperparams["learning_rate"]) if hyperparams['optimizer'] == "adam" else hyperparams["optimizer"], K.binary_crossentropy,
                               metrics=[metrics_class.precision_m, metrics_class.recall_m,
                                        metrics_class.f1_m, AUC()])
    model.summary()
    return model


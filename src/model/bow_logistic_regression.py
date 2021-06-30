import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LSTM, Concatenate, Lambda

from metrics import Metrics


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


if __name__ == '__main__':
    vector_size = 256
    model = build_bow_log_regression_model(bow_vector=vector_size, hyperparams={"threshold": 0.5, "optimizer": "adam"}, hyperparams_features=None)
    bow = np.random.random(size=(10, vector_size))
    print(model.predict(bow))
    # print(model.predict([[["a"], ["b"], ["c"]]]))

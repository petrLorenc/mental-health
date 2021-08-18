import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC

from train_utils.metrics import Metrics
from utils.default_config import DefaultHyperparameters

hyperparams = DefaultHyperparameters({
    "embeddings": "unigrams-features",
    "dense_units": 128
})

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
    model.compile(tf.optimizers.Adam(learning_rate=hyperparams["learning_rate"]) if "learning_rate" in hyperparams else hyperparams["optimizer"], K.binary_crossentropy,
                               metrics=[metrics_class.precision_m, metrics_class.recall_m,
                                        metrics_class.f1_m, AUC()])
    model.summary()
    return model


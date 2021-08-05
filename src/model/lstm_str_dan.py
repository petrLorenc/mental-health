import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LSTM, Concatenate

from train_utils.metrics import Metrics
from utils.default_config import DefaultHyperparametersSequence

hyperparams = DefaultHyperparametersSequence({
    "trainable_embeddings": False,
    "embeddings": "use-str"
})

hyperparams_features = {
    "module_url": "../resources/embeddings/use-4",
    "embedding_dim": 512
}


def build_lstm_with_str_input_dan(hyperparams, hyperparams_features):
    n_sentences = hyperparams['chunk_size']

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



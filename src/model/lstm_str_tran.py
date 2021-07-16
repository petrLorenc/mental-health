import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LSTM, Concatenate

from train_utils.metrics import Metrics

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
    "embeddings": "use-str",
    "positive_class_weight": 2,
    "maxlen": 50,
    "lstm_units_user": 100,
    "max_posts_per_user": 15,
    "batch_size": 16,

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
    "module_url": "../resources/embeddings/use-5",
    "embedding_dim": 512
}


def build_lstm_with_str_input_tran(hyperparams, hyperparams_features):
    n_sentences = hyperparams['max_posts_per_user']
    # print(n_sentences)
    # print(hyperparams_features['embedding_dim'])
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



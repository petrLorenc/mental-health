import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LSTM

from train_utils.metrics import Metrics

hyperparams = {
    "trainable_embeddings": False,
    "dropout": 0.1,
    "l2_dense": 0.00011,
    "l2_embeddings": 1e-07,
    "norm_momentum": 0.1,
    "ignore_layer": [],

    "epochs": 50,
    "embeddings": "distillbert-vector",
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
    # "module_url": "../resources/embeddings/distilbert-base-uncased",
    "embeddings_name": "vinai-bertweet",
    "embedding_dim": 768
}


def build_lstm_with_vector_input_distillbert(hyperparams, hyperparams_features):
    n_sentences = hyperparams['max_posts_per_user']

    _input = tf.keras.layers.Input(shape=(n_sentences, hyperparams_features['embedding_dim'],))
    x = tf.keras.layers.Masking(mask_value=0.)(_input)
    x = LSTM(hyperparams['lstm_units_user'], return_sequences=False)(x)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=_input, outputs=_output)
    metrics_class = Metrics(threshold=hyperparams['threshold'])
    model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                  metrics=[metrics_class.precision_m, metrics_class.recall_m,
                           metrics_class.f1_m, AUC()])
    model.summary()
    return model

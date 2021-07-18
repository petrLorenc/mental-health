import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LSTM

from train_utils.metrics import Metrics
from utils.default_config import DefaultHyperparametersSequence

hyperparams = DefaultHyperparametersSequence({
    "trainable_embeddings": False,
    "embeddings": "use-vector"
})

hyperparams_features = {
    "module_url": "../resources/embeddings/use-4",
    "embedding_dim": 512
}


def build_lstm_with_vector_input_dan(hyperparams, hyperparams_features):
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

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import GRU

from train_utils.metrics import Metrics
from utils.default_config import DefaultHyperparametersSequence

hyperparams = DefaultHyperparametersSequence({
    "trainable_embeddings": False,
    "embeddings": "use-str"
})

hyperparams_features = {
    "module_url": "../resources/embeddings/use-4"
}


def build_lstm_stateful_model(hyperparams, hyperparams_features):
    _input = tf.keras.layers.Input(shape=(1, hyperparams_features['embedding_dim'],), batch_size=hyperparams["batch_size"])
    x = GRU(512, stateful=True)(_input)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=_input, outputs=_output)
    metrics_class = Metrics(threshold=hyperparams['threshold'])
    model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                  metrics=[metrics_class.precision_m, metrics_class.recall_m,
                           metrics_class.f1_m, AUC()])
    model.summary()
    return model

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import LSTM, Dense, Flatten, RepeatVector, Permute, Activation, Multiply, Lambda, Input, Concatenate, Bidirectional, \
    BatchNormalization, LayerNormalization

from train_utils.metrics import Metrics
from utils.default_config import DefaultHyperparametersSequence

hyperparams = DefaultHyperparametersSequence({
    "trainable_embeddings": False,
    "embeddings": "use4.vstack",
    "lstm_units": 128,
    "positive_class_weight": 1,
    "chunk_size": 3,
    "dropout_rate": 0,
    "batch_size": 8
})

metrics_class = Metrics()


def build_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features):
    n_sentences = hyperparams['chunk_size']

    _input = tf.keras.layers.Input(shape=(n_sentences, hyperparams_features['embedding_dim'],))
    # x = tf.keras.layers.Masking(mask_value=0.)(_input)
    x = Bidirectional(LSTM(hyperparams['lstm_units'], return_sequences=False))(_input)
    x = tf.keras.layers.Dropout(hyperparams["dropout_rate"])(x)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=_input, outputs=_output)
    metrics_class.threshold = hyperparams['threshold']
    model.compile(tf.optimizers.Adam(learning_rate=hyperparams["learning_rate"]) if hyperparams['optimizer'] == "adam" else hyperparams["optimizer"], K.binary_crossentropy,
                  metrics=[metrics_class.precision_m, metrics_class.recall_m,
                           metrics_class.f1_m, AUC()])
    model.summary()
    return model


def build_attention_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features):
    n_sentences = hyperparams['chunk_size']

    _input = tf.keras.layers.Input(shape=(n_sentences, hyperparams_features['embedding_dim'],))
    # x = tf.keras.layers.Masking(mask_value=0.)(_input)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(512, activation="relu"))(_input)
    lstm_user_layers = LSTM(hyperparams['lstm_units'], return_sequences=True)(x)

    # attention mechanism
    attention_user_layer = Dense(1, activation='tanh', name='attention_user')
    attention_user = attention_user_layer(lstm_user_layers)
    attention_user = Flatten()(attention_user)
    attention_user_output = Activation('softmax', name="attention_output")(attention_user)
    attention_user = RepeatVector(hyperparams['lstm_units_user'])(attention_user_output)
    attention_user = Permute([2, 1])(attention_user)

    user_representation = Multiply()([lstm_user_layers, attention_user])
    # masking_func = lambda inputs, previous_mask: previous_mask

    user_representation = Lambda(lambda xin: K.sum(xin, axis=1))(user_representation)

    # user_representation = tf.keras.layers.Dropout(0.2)(user_representation)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(user_representation)

    model = tf.keras.Model(inputs=_input, outputs=_output)
    attention_model = tf.keras.Model(inputs=_input, outputs=attention_user_output)
    metrics_class.threshold = hyperparams['threshold']
    model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                  metrics=[metrics_class.precision_m, metrics_class.recall_m,
                           metrics_class.f1_m, AUC()])
    attention_model.compile(hyperparams['optimizer'], K.binary_crossentropy)
    model.summary()
    return model, attention_model


def build_attention_and_features_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features, additional_features_dim):
    n_sentences = hyperparams['chunk_size']

    _input = Input(shape=(n_sentences, hyperparams_features['embedding_dim'],))
    features_input = Input(shape=(n_sentences, additional_features_dim), name='additional_features')
    # x = Concatenate()([_input, features_input])
    x = tf.keras.layers.Masking(mask_value=0.)(_input)
    # emotions and pronouns
    lstm_user_layers = LSTM(hyperparams['lstm_units'], return_sequences=False)(x)
    x2 = BatchNormalization()(features_input)
    lstm_user_features = LSTM(32, return_sequences=False)(x2)

    # attention mechanism
    attention_user_layer = Dense(1, activation='tanh', name='attention_user')
    attention_user = attention_user_layer(lstm_user_layers)
    attention_user = Flatten()(attention_user)
    attention_user_output = Activation('softmax')(attention_user)
    attention_user = RepeatVector(hyperparams['lstm_units'])(attention_user_output)
    attention_user = Permute([2, 1])(attention_user)

    user_representation = Multiply()([lstm_user_layers, attention_user])
    user_representation = Lambda(lambda xin: K.sum(xin, axis=1),
                                 output_shape=(hyperparams['lstm_units_user'],))(user_representation)

    user_representation = Concatenate()([user_representation, lstm_user_features])

    _output = tf.keras.layers.Dense(1, activation="sigmoid")(user_representation)

    model = tf.keras.Model(inputs=[_input, features_input], outputs=_output)
    metrics_class.threshold = hyperparams['threshold']
    model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                  metrics=[metrics_class.precision_m, metrics_class.recall_m,
                           metrics_class.f1_m, AUC()])
    model.summary()

    attention_model = tf.keras.Model(inputs=[_input, features_input], outputs=attention_user_output)
    attention_model.compile(hyperparams['optimizer'], K.binary_crossentropy)

    return model, attention_model


def build_attention_and_aggregated_features_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features, additional_features_dim):
    n_sentences = hyperparams['chunk_size']

    _input = Input(shape=(n_sentences, hyperparams_features['embedding_dim'],))
    features_input = Input(shape=(additional_features_dim), name='additional_features')
    # x = Concatenate()([_input, features_input])
    # x = tf.keras.layers.Masking(mask_value=0.)(_input)
    # emotions and pronouns
    lstm_user_layers = LSTM(hyperparams['lstm_units'], return_sequences=False)(_input)

    # attention mechanism
    attention_user_layer = Dense(1, activation='tanh', name='attention_user')
    attention_user = attention_user_layer(lstm_user_layers)
    attention_user = Flatten()(attention_user)
    attention_user_output = Activation('softmax')(attention_user)
    attention_user = RepeatVector(hyperparams['lstm_units_user'])(attention_user_output)
    attention_user = Permute([2, 1])(attention_user)

    user_representation = Multiply()([lstm_user_layers, attention_user])
    user_representation = Lambda(lambda xin: K.sum(xin, axis=1),
                                 output_shape=(hyperparams['lstm_units'],))(user_representation)

    user_representation = Concatenate()([user_representation, features_input])
    user_representation = BatchNormalization()(user_representation)
    _output = tf.keras.layers.Dense(1, activation="sigmoid")(user_representation)
    metrics_class.threshold = hyperparams['threshold']
    model = tf.keras.Model(inputs=[_input, features_input], outputs=_output)
    model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                  metrics=[metrics_class.precision_m, metrics_class.recall_m,
                           metrics_class.f1_m, AUC()])
    model.summary()

    attention_model = tf.keras.Model(inputs=[_input, features_input], outputs=attention_user_output)
    attention_model.compile(hyperparams['optimizer'], K.binary_crossentropy)

    return model, attention_model

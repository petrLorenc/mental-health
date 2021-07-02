from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Lambda, BatchNormalization, TimeDistributed, \
    Bidirectional, Input, concatenate, Flatten, RepeatVector, Activation, Multiply, Permute, \
    Conv1D, GlobalMaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import AUC
from metrics import Metrics
from resource_loading import load_embeddings, load_dict_from_file

import numpy as np

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
    "embeddings": "glove",
    "positive_class_weight": 2,
    "maxlen": 50,
    "lstm_units": 64,
    "lstm_units_user": 64,
    "max_posts_per_user": 15,
    "batch_size": 64,

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
    "vocabulary_path": "../resources/generated/vocab_20000_erisk.txt",
    "nrc_lexicon_path": "../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "liwc_path": "../resources/liwc.dic",
    "stopwords_path": "../resources/stopwords.txt",
    "embeddings_path": "../resources/embeddings/glove.840B.300d.txt",
    "liwc_words_cached": "../resources/generated/liwc_categories_for_vocabulary_erisk_clpsych_stop_20K.pkl"
}


def build_hierarchical_model(hyperparams, hyperparams_features,
                             emotions_dim, stopwords_list_dim, liwc_categories_dim, word_embedding_type,
                             ignore_layer=[]):
    vocabulary = load_dict_from_file(hyperparams_features["vocabulary_path"])

    if word_embedding_type == "random":
        # dummy embedding matrix - ONLY FOR TESTING
        embedding_matrix = np.random.random((len(vocabulary), hyperparams_features['embedding_dim'])) - 0.5
    elif word_embedding_type == "glove":
        # real embedding matrix
        embedding_matrix = load_embeddings(embeddings_path=hyperparams_features['embeddings_path'],
                                           embedding_dim=hyperparams_features['embedding_dim'],
                                           vocabulary=vocabulary)
    else:
        raise NotImplementedError(f"Embeddings {word_embedding_type} not supported yet")

    # Post/sentence representation - word sequence
    tokens_features = Input(shape=(hyperparams['maxlen'],), name='word_seq')
    embedding_layer = Embedding(len(vocabulary),
                                hyperparams_features['embedding_dim'],
                                input_length=hyperparams['maxlen'],
                                embeddings_regularizer=regularizers.l2(hyperparams['l2_embeddings']),
                                weights=[embedding_matrix],
                                trainable=hyperparams['trainable_embeddings'],
                                name='embeddings_layer',
                                mask_zero=True)(tokens_features)
    embedding_layer = Dropout(hyperparams['dropout'], name='embedding_dropout')(embedding_layer)

    lstm_layers = LSTM(hyperparams['lstm_units'],
                       return_sequences='attention' not in ignore_layer,
                       name='LSTM_layer')(embedding_layer)

    # Attention
    if 'attention' not in ignore_layer:
        attention_layer = Dense(1, activation='tanh', name='attention')
        attention = attention_layer(lstm_layers)
        attention = Flatten()(attention)
        attention_output = Activation('softmax')(attention)
        attention = RepeatVector(hyperparams['lstm_units'])(attention_output)
        attention = Permute([2, 1])(attention)

        sent_representation = Multiply()([lstm_layers, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1),
                                     output_shape=(hyperparams['lstm_units'],)
                                     )(sent_representation)
    else:
        sent_representation = lstm_layers

    if 'batchnorm' not in ignore_layer:
        sent_representation = BatchNormalization(axis=1, momentum=hyperparams['norm_momentum'], name='sent_repr_norm')(sent_representation)
    sent_representation = Dropout(hyperparams['dropout'], name='sent_repr_dropout')(sent_representation)

    # Other features
    numerical_features_history = Input(shape=(
        hyperparams['max_posts_per_user'],
        emotions_dim + 1 + liwc_categories_dim
    ), name='numeric_input_hist')  # emotions and pronouns
    sparse_features_history = Input(shape=(
        hyperparams['max_posts_per_user'],
        stopwords_list_dim
    ), name='sparse_input_hist')  # stopwords

    posts_history_input = Input(shape=(hyperparams['max_posts_per_user'],
                                       hyperparams['maxlen']
                                       ), name='hierarchical_word_seq_input')

    # Hierarchy
    sentEncoder = Model(inputs=tokens_features,
                        outputs=sent_representation)
    sentEncoder.summary()

    user_encoder = TimeDistributed(sentEncoder, name='user_encoder')(posts_history_input)

    dense_layer_sparse = Dense(units=hyperparams['dense_bow_units'],
                               name='sparse_feat_dense_layer', activation='relu',
                               kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                               )
    dense_layer_sparse_user = TimeDistributed(dense_layer_sparse,
                                              name='sparse_dense_layer_user')(sparse_features_history)

    dense_layer_numerical = Dense(units=hyperparams['dense_numerical_units'],
                                  name='numerical_feat_dense_layer', activation='relu',
                                  kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                                  )
    dense_layer_numerical_user = TimeDistributed(dense_layer_numerical,
                                                 name='numerical_dense_layer_user')(numerical_features_history)

    # Concatenate features
    if 'batchnorm' not in ignore_layer:
        dense_layer_numerical_user = BatchNormalization(axis=1, momentum=hyperparams['norm_momentum'],
                                                        name='numerical_features_norm')(dense_layer_numerical_user)
        dense_layer_sparse_user = BatchNormalization(axis=1, momentum=hyperparams['norm_momentum'],
                                                     name='sparse_features_norm')(dense_layer_sparse_user)
    all_layers = {
        'user_encoded': user_encoder,

        'numerical_dense_layer': dense_layer_numerical_user,

        'sparse_feat_dense_layer': dense_layer_sparse_user,
    }

    layers_to_merge = [l for n, l in all_layers.items() if n not in ignore_layer]
    if len(layers_to_merge) == 1:
        merged_layers = layers_to_merge[0]
    else:
        merged_layers = concatenate(layers_to_merge)

    lstm_user_layers = LSTM(hyperparams['lstm_units_user'],
                            return_sequences='attention_user' not in ignore_layer,
                            name='LSTM_layer_user')(merged_layers)

    # Attention
    if 'attention_user' not in ignore_layer:
        attention_user_layer = Dense(1, activation='tanh', name='attention_user')
        attention_user = attention_user_layer(lstm_user_layers)
        attention_user = Flatten()(attention_user)
        attention_user_output = Activation('softmax')(attention_user)
        attention_user = RepeatVector(hyperparams['lstm_units_user'])(attention_user_output)
        attention_user = Permute([2, 1])(attention_user)

        user_representation = Multiply()([lstm_user_layers, attention_user])
        user_representation = Lambda(lambda xin: K.sum(xin, axis=1),
                                     output_shape=(hyperparams['lstm_units_user'],))(user_representation)

    else:
        user_representation = lstm_user_layers

    user_representation = Dropout(hyperparams['dropout'], name='user_repr_dropout')(user_representation)

    if hyperparams['dense_user_units']:
        user_representation = Dense(units=hyperparams['dense_user_units'], activation='relu',
                                    name='dense_user_representation')(user_representation)

    output_layer = Dense(1, activation='sigmoid',
                         name='output_layer',
                         kernel_regularizer=regularizers.l2(hyperparams['l2_dense'])
                         )(user_representation)

    hierarchical_model = Model(inputs=[posts_history_input,
                                       numerical_features_history, sparse_features_history,
                                       ],
                               outputs=output_layer)

    metrics_class = Metrics(threshold=hyperparams['threshold'])
    hierarchical_model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                               metrics=[metrics_class.precision_m, metrics_class.recall_m,
                                        metrics_class.f1_m, AUC()])

    return hierarchical_model


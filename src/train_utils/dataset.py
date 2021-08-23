from utils.logger import logger

import json

from loader.DataGeneratorFeatures import DataGeneratorHierarchical
from loader.DataGeneratorPrecomputedWordsVector import DataGeneratorHierarchicalPrecomputed
from loader.DataGeneratorStr import DataGeneratorStr
# from loader.DataGeneratorUniGrams import DataGeneratorUnigrams
from loader.DataGeneratorUniGramsChunked import DataGeneratorUnigrams
from loader.DataGeneratorUniGramsFeatures import DataGeneratorUnigramsFeatures
# from loader.DataGeneratorBiGrams import DataGeneratorBiGrams
from loader.DataGeneratorBiGramsChunked import DataGeneratorBiGrams
from loader.DataGeneratorTensorFlowHubVector import DataGeneratorTensorFlowHubVector
from loader.DataGeneratorStateful import DataGeneratorStateful
from loader.DataGeneratorPrecomputedVectorSequence import DataGeneratorPrecomputedVectorSequence
from loader.DataGeneratorPrecomputedVectorAggregated import DataGeneratorPrecomputedVectorAggregated
from loader.DataGeneratorPrecomputedVectorSequenceMultiple import DataGeneratorPrecomputedGroupOfVectorsSequence


def initialize_datasets_hierarchical(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorHierarchical(user_level_data, subjects_split, set_type='train',
                                                     hyperparams_features=hyperparams_features,
                                                     max_seq_len=hyperparams['max_seq_len'], batch_size=hyperparams['batch_size'],
                                                     chunk_size=hyperparams['chunk_size'],
                                                     shuffle=False, data_generator_id="train")

    data_generator_valid = DataGeneratorHierarchical(user_level_data, subjects_split, set_type="valid",
                                                     hyperparams_features=hyperparams_features,
                                                     max_seq_len=hyperparams['max_seq_len'], batch_size=hyperparams['batch_size'],
                                                     chunk_size=hyperparams['chunk_size'],
                                                     shuffle=False, data_generator_id="valid")

    data_generator_test = DataGeneratorHierarchical(user_level_data, subjects_split, set_type="test",
                                                    hyperparams_features=hyperparams_features,
                                                    max_seq_len=hyperparams['max_seq_len'], batch_size=1,
                                                    chunk_size=hyperparams['chunk_size'],
                                                    shuffle=False, data_generator_id="test")
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_hierarchical_precomputed(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorHierarchicalPrecomputed(user_level_data, subjects_split, set_type='train',
                                                                hyperparams_features=hyperparams_features,
                                                                max_seq_len=hyperparams['max_seq_len'], batch_size=hyperparams['batch_size'],
                                                                chunk_size=hyperparams['chunk_size'],
                                                                shuffle=False, data_generator_id="train",
                                                                embedding_dimension=hyperparams_features["embedding_dim"],
                                                                precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                                feature_extraction_name=hyperparams_features["embeddings_name"],
                                                                emotions_dim=hyperparams["emotions_dim"],
                                                                liwc_categories_dim=hyperparams["liwc_categories_dim"],
                                                                stopwords_list_dim=hyperparams["stopwords_dim"])

    data_generator_valid = DataGeneratorHierarchicalPrecomputed(user_level_data, subjects_split, set_type="valid",
                                                                hyperparams_features=hyperparams_features,
                                                                max_seq_len=hyperparams['max_seq_len'], batch_size=hyperparams['batch_size'],
                                                                chunk_size=hyperparams['chunk_size'],
                                                                shuffle=False, data_generator_id="valid",
                                                                embedding_dimension=hyperparams_features["embedding_dim"],
                                                                precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                                feature_extraction_name=hyperparams_features["embeddings_name"],
                                                                emotions_dim=hyperparams["emotions_dim"],
                                                                liwc_categories_dim=hyperparams["liwc_categories_dim"],
                                                                stopwords_list_dim=hyperparams["stopwords_dim"])

    data_generator_test = DataGeneratorHierarchicalPrecomputed(user_level_data, subjects_split, set_type="test",
                                                               hyperparams_features=hyperparams_features,
                                                               max_seq_len=hyperparams['max_seq_len'], batch_size=1,
                                                               chunk_size=hyperparams['chunk_size'],
                                                               shuffle=False, data_generator_id="test",
                                                               embedding_dimension=hyperparams_features["embedding_dim"],
                                                               precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                               feature_extraction_name=hyperparams_features["embeddings_name"],
                                                               emotions_dim=hyperparams["emotions_dim"],
                                                               liwc_categories_dim=hyperparams["liwc_categories_dim"],
                                                               stopwords_list_dim=hyperparams["stopwords_dim"])
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_str(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorStr(user_level_data, subjects_split, set_type='train',
                                            max_seq_len=None, batch_size=hyperparams['batch_size'],
                                            chunk_size=hyperparams['chunk_size'],
                                            shuffle=True, data_generator_id="train")

    data_generator_valid = DataGeneratorStr(user_level_data, subjects_split, set_type="valid",
                                            max_seq_len=None, batch_size=hyperparams['batch_size'],
                                            chunk_size=hyperparams['chunk_size'],
                                            shuffle=False, data_generator_id="valid")

    data_generator_test = DataGeneratorStr(user_level_data, subjects_split, set_type="test",
                                           max_seq_len=None, batch_size=1,
                                           chunk_size=hyperparams['chunk_size'],
                                           shuffle=False, data_generator_id="test")
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_tensorflowhub_vector(user_level_data, subjects_split, hyperparams, hyperparams_features):
    import tensorflow_hub as hub
    vectorizer = hub.load(hyperparams_features["module_url"])
    data_generator_train = DataGeneratorTensorFlowHubVector(user_level_data=user_level_data, subjects_split=subjects_split, set_type='train',
                                                            max_seq_len=hyperparams['max_seq_len'], batch_size=hyperparams['batch_size'],
                                                            chunk_size=hyperparams['chunk_size'],
                                                            shuffle=False, data_generator_id="train", vectorizer=vectorizer,
                                                            embedding_dimension=hyperparams_features["embedding_dim"])

    data_generator_valid = DataGeneratorTensorFlowHubVector(user_level_data=user_level_data, subjects_split=subjects_split, set_type="valid",
                                                            max_seq_len=hyperparams['max_seq_len'], batch_size=hyperparams['batch_size'],
                                                            chunk_size=hyperparams['chunk_size'],
                                                            shuffle=False, data_generator_id="valid", vectorizer=vectorizer,
                                                            embedding_dimension=hyperparams_features["embedding_dim"])

    data_generator_test = DataGeneratorTensorFlowHubVector(user_level_data=user_level_data, subjects_split=subjects_split, set_type="test",
                                                           max_seq_len=hyperparams['max_seq_len'], batch_size=1,
                                                           chunk_size=hyperparams['chunk_size'],
                                                           shuffle=False, data_generator_id="test", vectorizer=vectorizer,
                                                           embedding_dimension=hyperparams_features["embedding_dim"])
    return data_generator_train, data_generator_valid, data_generator_test


import numpy as np


def initialize_datasets_precomputed_vector_aggregated(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorPrecomputedVectorAggregated(user_level_data=user_level_data, subjects_split=subjects_split, set_type='train',
                                                                    max_seq_len=hyperparams['max_seq_len'], batch_size=hyperparams['batch_size'],
                                                                    chunk_size=hyperparams['chunk_size'],
                                                                    shuffle=False, data_generator_id="train",
                                                                    embedding_dimension=hyperparams_features["embedding_dim"],
                                                                    precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                                    feature_extraction_name=hyperparams_features["embeddings_name"])

    data_generator_valid = DataGeneratorPrecomputedVectorAggregated(user_level_data=user_level_data, subjects_split=subjects_split, set_type="valid",
                                                                    max_seq_len=hyperparams['max_seq_len'], batch_size=hyperparams['batch_size'],
                                                                    chunk_size=hyperparams['chunk_size'],
                                                                    shuffle=False, data_generator_id="valid",
                                                                    embedding_dimension=hyperparams_features["embedding_dim"],
                                                                    precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                                    feature_extraction_name=hyperparams_features["embeddings_name"])

    data_generator_test = DataGeneratorPrecomputedVectorAggregated(user_level_data=user_level_data, subjects_split=subjects_split, set_type="test",
                                                                   max_seq_len=hyperparams['max_seq_len'], batch_size=1,
                                                                   chunk_size=hyperparams['chunk_size'],
                                                                   shuffle=False, data_generator_id="test",
                                                                   embedding_dimension=hyperparams_features["embedding_dim"],
                                                                   precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                                   feature_extraction_name=hyperparams_features["embeddings_name"])
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_precomputed_vector_sequence(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorPrecomputedVectorSequence(user_level_data=user_level_data, subjects_split=subjects_split, set_type='train',
                                                                  max_seq_len=0, batch_size=hyperparams['batch_size'],
                                                                  chunk_size=hyperparams['chunk_size'],
                                                                  shuffle=False, data_generator_id="train",
                                                                  embedding_dimension=hyperparams_features["embedding_dim"],
                                                                  precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                                  feature_extraction_name=hyperparams_features["embeddings_name"])

    data_generator_valid = DataGeneratorPrecomputedVectorSequence(user_level_data=user_level_data, subjects_split=subjects_split, set_type="valid",
                                                                  max_seq_len=0, batch_size=1,
                                                                  chunk_size=hyperparams['chunk_size'],
                                                                  shuffle=False, data_generator_id="valid",
                                                                  embedding_dimension=hyperparams_features["embedding_dim"],
                                                                  precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                                  feature_extraction_name=hyperparams_features["embeddings_name"])

    data_generator_test = DataGeneratorPrecomputedVectorSequence(user_level_data=user_level_data, subjects_split=subjects_split, set_type="test",
                                                                 max_seq_len=0, batch_size=1,
                                                                 chunk_size=hyperparams['chunk_size'],
                                                                 shuffle=False, data_generator_id="test",
                                                                 embedding_dimension=hyperparams_features["embedding_dim"],
                                                                 precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                                 feature_extraction_name=hyperparams_features["embeddings_name"])
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_precomputed_group_of_vectors_sequence(user_level_data, subjects_split, hyperparams, hyperparams_features):
    other_features = ["emotions",
                      "liwc",
                      "pronouns",
                      "stopwords"]
    data_generator_train = DataGeneratorPrecomputedGroupOfVectorsSequence(user_level_data=user_level_data, subjects_split=subjects_split,
                                                                          set_type='train',
                                                                          max_seq_len=hyperparams['max_seq_len'],
                                                                          batch_size=hyperparams['batch_size'],
                                                                          chunk_size=hyperparams['chunk_size'],
                                                                          shuffle=False, data_generator_id="train",
                                                                          embedding_dimension=hyperparams_features["embedding_dim"],
                                                                          precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                                          embedding_name=hyperparams_features["embeddings_name"],
                                                                          other_features=other_features, aggregate_other_features=True)

    data_generator_valid = DataGeneratorPrecomputedGroupOfVectorsSequence(user_level_data=user_level_data, subjects_split=subjects_split,
                                                                          set_type="valid",
                                                                          max_seq_len=hyperparams['max_seq_len'],
                                                                          batch_size=hyperparams['batch_size'],
                                                                          chunk_size=hyperparams['chunk_size'],
                                                                          shuffle=False, data_generator_id="valid",
                                                                          embedding_dimension=hyperparams_features["embedding_dim"],
                                                                          precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                                          embedding_name=hyperparams_features["embeddings_name"],
                                                                          other_features=other_features, aggregate_other_features=True)

    data_generator_test = DataGeneratorPrecomputedGroupOfVectorsSequence(user_level_data=user_level_data, subjects_split=subjects_split,
                                                                         set_type="test",
                                                                         max_seq_len=hyperparams['max_seq_len'], batch_size=1,
                                                                         chunk_size=hyperparams['chunk_size'],
                                                                         shuffle=False, data_generator_id="test",
                                                                         embedding_dimension=hyperparams_features["embedding_dim"],
                                                                         precomputed_vectors_path=hyperparams_features["precomputed_vectors_path"],
                                                                         embedding_name=hyperparams_features["embeddings_name"],
                                                                         other_features=other_features, aggregate_other_features=True)
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_unigrams(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorUnigrams(user_level_data, subjects_split, set_type='train',
                                                 hyperparams_features=hyperparams_features,
                                                 batch_size=hyperparams['batch_size'], keep_first_batches=False,
                                                 data_generator_id="train", chunk_size=hyperparams['chunk_size'])

    data_generator_valid = DataGeneratorUnigrams(user_level_data, subjects_split, set_type="valid",
                                                 hyperparams_features=hyperparams_features,
                                                 batch_size=hyperparams['batch_size'], keep_first_batches=False,
                                                 vectorizer=data_generator_train.vectorizer,
                                                 data_generator_id="valid", chunk_size=hyperparams['chunk_size'])

    data_generator_test = DataGeneratorUnigrams(user_level_data, subjects_split, set_type="test",
                                                hyperparams_features=hyperparams_features,
                                                batch_size=hyperparams['batch_size'], keep_first_batches=False,
                                                vectorizer=data_generator_train.vectorizer,
                                                data_generator_id="test", chunk_size=hyperparams['chunk_size'])
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_unigrams_features(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorUnigramsFeatures(user_level_data, subjects_split, set_type='train',
                                                         hyperparams_features=hyperparams_features,
                                                         batch_size=1,
                                                         data_generator_id="train")

    data_generator_valid = DataGeneratorUnigramsFeatures(user_level_data, subjects_split, set_type="valid",
                                                         hyperparams_features=hyperparams_features,
                                                         batch_size=1,
                                                         vectorizer=data_generator_train.vectorizer,
                                                         data_generator_id="valid")

    data_generator_test = DataGeneratorUnigramsFeatures(user_level_data, subjects_split, set_type="test",
                                                        hyperparams_features=hyperparams_features,
                                                        batch_size=1,
                                                        vectorizer=data_generator_train.vectorizer,
                                                        data_generator_id="test")
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_bigrams(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorBiGrams(user_level_data, subjects_split, set_type='train',
                                                hyperparams_features=hyperparams_features,
                                                batch_size=hyperparams['batch_size'],
                                                data_generator_id="train", chunk_size=hyperparams['chunk_size'])

    data_generator_valid = DataGeneratorBiGrams(user_level_data, subjects_split, set_type="valid",
                                                hyperparams_features=hyperparams_features,
                                                batch_size=hyperparams['batch_size'],
                                                vectorizer=data_generator_train.vectorizer,
                                                data_generator_id="valid", chunk_size=hyperparams['chunk_size'])

    data_generator_test = DataGeneratorBiGrams(user_level_data, subjects_split, set_type="test",
                                               hyperparams_features=hyperparams_features,
                                               batch_size=hyperparams['batch_size'],
                                               vectorizer=data_generator_train.vectorizer,
                                               data_generator_id="test", chunk_size=hyperparams['chunk_size'])
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_stateful(user_level_data, subjects_split, hyperparams, hyperparams_features):
    import tensorflow_hub as hub
    vectorizer = hub.load(hyperparams_features["module_url"])
    data_generator_train = DataGeneratorStateful(user_level_data, subjects_split, set_type='train',
                                                 batch_size=1,
                                                 data_generator_id="train", vectorizer=vectorizer)

    data_generator_valid = DataGeneratorStateful(user_level_data, subjects_split, set_type="valid",
                                                 batch_size=1,
                                                 vectorizer=data_generator_train.vectorizer,
                                                 data_generator_id="valid")

    data_generator_test = DataGeneratorStateful(user_level_data, subjects_split, set_type="test",
                                                batch_size=1,
                                                vectorizer=data_generator_train.vectorizer,
                                                data_generator_id="test")
    return data_generator_train, data_generator_valid, data_generator_test

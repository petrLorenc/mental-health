import json

from loader.DataGeneratorFeatures import DataGeneratorHierarchical
from loader.DataGeneratorStr import DataGeneratorStr
from loader.DataGeneratorUniGrams import DataGeneratorUnigrams
from loader.DataGeneratorUniGramsFeatures import DataGeneratorUnigramsFeatures
from loader.DataGeneratorBiGrams import DataGeneratorBiGrams
from loader.DataGeneratorVector_USE import DataGeneratorUSEVector
from loader.DataGeneratorStateful import DataGeneratorStateful
from loader.DataGeneratorVector_DistillBERT import DataGeneratorVectorDistilBERT


def initialize_datasets_hierarchical(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorHierarchical(user_level_data, subjects_split, set_type='train',
                                                     hyperparams_features=hyperparams_features,
                                                     seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                     max_posts_per_user=hyperparams['max_posts_per_user'],
                                                     shuffle=False, data_generator_id="train")

    data_generator_valid = DataGeneratorHierarchical(user_level_data, subjects_split, set_type="valid",
                                                     hyperparams_features=hyperparams_features,
                                                     seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                     max_posts_per_user=hyperparams['max_posts_per_user'],
                                                     shuffle=False, data_generator_id="valid")

    data_generator_test = DataGeneratorHierarchical(user_level_data, subjects_split, set_type="test",
                                                    hyperparams_features=hyperparams_features,
                                                    seq_len=hyperparams['maxlen'], batch_size=1,
                                                    max_posts_per_user=hyperparams['max_posts_per_user'],
                                                    shuffle=False, data_generator_id="test")
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_str(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorStr(user_level_data, subjects_split, set_type='train',
                                            seq_len=None, batch_size=hyperparams['batch_size'],
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            shuffle=True, data_generator_id="train")

    data_generator_valid = DataGeneratorStr(user_level_data, subjects_split, set_type="valid",
                                            seq_len=None, batch_size=hyperparams['batch_size'],
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            shuffle=False, data_generator_id="valid")

    data_generator_test = DataGeneratorStr(user_level_data, subjects_split, set_type="test",
                                           seq_len=None, batch_size=1,
                                           max_posts_per_user=hyperparams['max_posts_per_user'],
                                           shuffle=False, data_generator_id="test")
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_use_vector(user_level_data, subjects_split, hyperparams, hyperparams_features):
    import tensorflow_hub as hub
    vectorizer = hub.load(hyperparams_features["module_url"])
    data_generator_train = DataGeneratorUSEVector(user_level_data=user_level_data, subjects_split=subjects_split, set_type='train',
                                                  seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                  max_posts_per_user=hyperparams['max_posts_per_user'],
                                                  shuffle=False, data_generator_id="train", vectorizer=vectorizer,
                                                  embedding_dimension=hyperparams_features["embedding_dim"])

    data_generator_valid = DataGeneratorUSEVector(user_level_data=user_level_data, subjects_split=subjects_split, set_type="valid",
                                                  seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                  max_posts_per_user=hyperparams['max_posts_per_user'],
                                                  shuffle=False, data_generator_id="valid", vectorizer=vectorizer,
                                                  embedding_dimension=hyperparams_features["embedding_dim"])

    data_generator_test = DataGeneratorUSEVector(user_level_data=user_level_data, subjects_split=subjects_split, set_type="test",
                                                 seq_len=hyperparams['maxlen'], batch_size=1,
                                                 max_posts_per_user=hyperparams['max_posts_per_user'],
                                                 shuffle=False, data_generator_id="test", vectorizer=vectorizer,
                                                 embedding_dimension=hyperparams_features["embedding_dim"])
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_distillbert_vector(user_level_data, subjects_split, hyperparams, hyperparams_features):
    from transformers import AutoTokenizer, pipeline, TFDistilBertModel
    model = TFDistilBertModel.from_pretrained(hyperparams_features["module_url"])
    tokenizer = AutoTokenizer.from_pretrained(hyperparams_features["module_url"])
    vectorizer = pipeline('feature-extraction', model=model,
                          tokenizer=tokenizer)

    data_generator_train = DataGeneratorVectorDistilBERT(user_level_data=user_level_data, subjects_split=subjects_split, set_type='train',
                                                         seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                         max_posts_per_user=hyperparams['max_posts_per_user'],
                                                         shuffle=False, data_generator_id="train", vectorizer=vectorizer,
                                                         embedding_dimension=hyperparams_features["embedding_dim"])

    data_generator_valid = DataGeneratorVectorDistilBERT(user_level_data=user_level_data, subjects_split=subjects_split, set_type="valid",
                                                         seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                         max_posts_per_user=hyperparams['max_posts_per_user'],
                                                         shuffle=False, data_generator_id="valid", vectorizer=vectorizer,
                                                         embedding_dimension=hyperparams_features["embedding_dim"])

    data_generator_test = DataGeneratorVectorDistilBERT(user_level_data=user_level_data, subjects_split=subjects_split, set_type="test",
                                                        seq_len=hyperparams['maxlen'], batch_size=1,
                                                        max_posts_per_user=hyperparams['max_posts_per_user'],
                                                        shuffle=False, data_generator_id="test", vectorizer=vectorizer,
                                                        embedding_dimension=hyperparams_features["embedding_dim"])
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_unigrams(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorUnigrams(user_level_data, subjects_split, set_type='train',
                                                 hyperparams_features=hyperparams_features,
                                                 batch_size=1,
                                                 data_generator_id="train")

    data_generator_valid = DataGeneratorUnigrams(user_level_data, subjects_split, set_type="valid",
                                                 hyperparams_features=hyperparams_features,
                                                 batch_size=1,
                                                 vectorizer=data_generator_train.vectorizer,
                                                 data_generator_id="valid")

    data_generator_test = DataGeneratorUnigrams(user_level_data, subjects_split, set_type="test",
                                                hyperparams_features=hyperparams_features,
                                                batch_size=1,
                                                vectorizer=data_generator_train.vectorizer,
                                                data_generator_id="test")
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
                                                batch_size=1,
                                                data_generator_id="train")

    data_generator_valid = DataGeneratorBiGrams(user_level_data, subjects_split, set_type="valid",
                                                hyperparams_features=hyperparams_features,
                                                batch_size=1,
                                                vectorizer=data_generator_train.vectorizer,
                                                data_generator_id="valid")

    data_generator_test = DataGeneratorBiGrams(user_level_data, subjects_split, set_type="test",
                                               hyperparams_features=hyperparams_features,
                                               batch_size=1,
                                               vectorizer=data_generator_train.vectorizer,
                                               data_generator_id="test")
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

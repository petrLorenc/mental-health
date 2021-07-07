import json

from loader.DataGeneratorFeatures import DataGeneratorHierarchical
from loader.DataGeneratorRaw import DataGeneratorRaw
from loader.DataGeneratorBOW import DataGeneratorBow
from loader.DataGeneratorUSE import DataGeneratorUSE
from loader.DataGeneratorStateful import DataGeneratorStateful


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


def initialize_datasets_raw(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorRaw(user_level_data, subjects_split, set_type='train',
                                            seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            shuffle=False, data_generator_id="train")

    data_generator_valid = DataGeneratorRaw(user_level_data, subjects_split, set_type="valid",
                                            seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            shuffle=False, data_generator_id="valid")

    data_generator_test = DataGeneratorRaw(user_level_data, subjects_split, set_type="test",
                                           seq_len=hyperparams['maxlen'], batch_size=1,
                                           max_posts_per_user=hyperparams['max_posts_per_user'],
                                           shuffle=False, data_generator_id="test")
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_use(user_level_data, subjects_split, hyperparams, hyperparams_features):
    import tensorflow_hub as hub
    vectorizer = hub.load(hyperparams_features["module_url"])
    data_generator_train = DataGeneratorUSE(user_level_data=user_level_data, subjects_split=subjects_split, set_type='train',
                                            seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            shuffle=False, data_generator_id="train", vectorizer=vectorizer)

    data_generator_valid = DataGeneratorUSE(user_level_data=user_level_data, subjects_split=subjects_split, set_type="valid",
                                            seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            shuffle=False, data_generator_id="valid", vectorizer=vectorizer)

    data_generator_test = DataGeneratorUSE(user_level_data=user_level_data, subjects_split=subjects_split, set_type="test",
                                           seq_len=hyperparams['maxlen'], batch_size=1,
                                           max_posts_per_user=hyperparams['max_posts_per_user'],
                                           shuffle=False, data_generator_id="test", vectorizer=vectorizer)
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_bow(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorBow(user_level_data, subjects_split, set_type='train',
                                            hyperparams_features=hyperparams_features,
                                            batch_size=1,
                                            data_generator_id="train")

    data_generator_valid = DataGeneratorBow(user_level_data, subjects_split, set_type="valid",
                                            hyperparams_features=hyperparams_features,
                                            batch_size=1,
                                            vectorizer=data_generator_train.vectorizer,
                                            data_generator_id="valid")

    data_generator_test = DataGeneratorBow(user_level_data, subjects_split, set_type="test",
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

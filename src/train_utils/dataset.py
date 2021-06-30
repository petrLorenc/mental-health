import json

from loader.DataGeneratorFeatures import DataGeneratorHierarchical
from loader.DataGeneratorRaw import DataGeneratorRaw
from loader.DataGeneratorBOW import DataGeneratorBow


def initialize_datasets_hierarchical(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorHierarchical(user_level_data, subjects_split, set_type='train',
                                                     hyperparams_features=hyperparams_features,
                                                     seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                     max_posts_per_user=hyperparams['max_posts_per_user'],
                                                     shuffle=False)

    data_generator_valid = DataGeneratorHierarchical(user_level_data, subjects_split, set_type="valid",
                                                     hyperparams_features=hyperparams_features,
                                                     seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                     max_posts_per_user=hyperparams['max_posts_per_user'],
                                                     shuffle=False)

    data_generator_test = DataGeneratorHierarchical(user_level_data, subjects_split, set_type="test",
                                                    hyperparams_features=hyperparams_features,
                                                    seq_len=hyperparams['maxlen'], batch_size=1,
                                                    max_posts_per_user=hyperparams['max_posts_per_user'],
                                                    shuffle=False)
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_raw(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorRaw(user_level_data, subjects_split, set_type='train',
                                            hyperparams_features=hyperparams_features,
                                            seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            shuffle=False)

    data_generator_valid = DataGeneratorRaw(user_level_data, subjects_split, set_type="valid",
                                            hyperparams_features=hyperparams_features,
                                            seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            shuffle=False)

    data_generator_test = DataGeneratorRaw(user_level_data, subjects_split, set_type="test",
                                           hyperparams_features=hyperparams_features,
                                           seq_len=hyperparams['maxlen'], batch_size=1,
                                           max_posts_per_user=hyperparams['max_posts_per_user'],
                                           shuffle=False)
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_bow(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGeneratorBow(user_level_data, subjects_split, set_type='train',
                                            hyperparams_features=hyperparams_features,
                                            seq_len=hyperparams['maxlen'], batch_size=1,
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            shuffle=False)

    data_generator_valid = DataGeneratorBow(user_level_data, subjects_split, set_type="valid",
                                            hyperparams_features=hyperparams_features,
                                            seq_len=hyperparams['maxlen'], batch_size=1,
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            shuffle=False)

    data_generator_test = DataGeneratorBow(user_level_data, subjects_split, set_type="test",
                                           hyperparams_features=hyperparams_features,
                                           seq_len=hyperparams['maxlen'], batch_size=1,
                                           max_posts_per_user=hyperparams['max_posts_per_user'],
                                           shuffle=False)
    return data_generator_train, data_generator_valid, data_generator_test


import json

from loader.EriskDataGeneratorFeatures import EriskDataGenerator
from loader.EriskDataGeneratorRaw import EriskDataGeneratorRaw
from loader.EriskDataGeneratorBOW import EriskDataGeneratorBow

from loader.DAICDataGeneratorFeatures import DAICDataGenerator
from loader.DAICDataGeneratorRaw import DAICDataGeneratorRaw
from loader.DAICDataGeneratorBOW import DAICDataGeneratorBoW


def initialize_datasets_erisk(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = EriskDataGenerator(user_level_data, subjects_split, set_type='train',
                                              hyperparams_features=hyperparams_features,
                                              seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                              max_posts_per_user=hyperparams['max_posts_per_user'],
                                              shuffle=False)

    data_generator_valid = EriskDataGenerator(user_level_data, subjects_split, set_type="valid",
                                              hyperparams_features=hyperparams_features,
                                              seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                              max_posts_per_user=hyperparams['max_posts_per_user'],
                                              shuffle=False)

    data_generator_test = EriskDataGenerator(user_level_data, subjects_split, set_type="test",
                                             hyperparams_features=hyperparams_features,
                                             seq_len=hyperparams['maxlen'], batch_size=1,
                                             max_posts_per_user=hyperparams['max_posts_per_user'],
                                             shuffle=False)
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_erisk_raw(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = EriskDataGeneratorRaw(user_level_data, subjects_split, set_type='train',
                                                 hyperparams_features=hyperparams_features,
                                                 seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                 max_posts_per_user=hyperparams['max_posts_per_user'],
                                                 shuffle=False)

    data_generator_valid = EriskDataGeneratorRaw(user_level_data, subjects_split, set_type="valid",
                                                 hyperparams_features=hyperparams_features,
                                                 seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                 max_posts_per_user=hyperparams['max_posts_per_user'],
                                                 shuffle=False)

    data_generator_test = EriskDataGeneratorRaw(user_level_data, subjects_split, set_type="test",
                                                hyperparams_features=hyperparams_features,
                                                seq_len=hyperparams['maxlen'], batch_size=1,
                                                max_posts_per_user=hyperparams['max_posts_per_user'],
                                                shuffle=False)
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_erisk_bow(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = EriskDataGeneratorBow(user_level_data, subjects_split, set_type='train',
                                                 hyperparams_features=hyperparams_features,
                                                 seq_len=hyperparams['maxlen'], batch_size=1,
                                                 max_posts_per_user=hyperparams['max_posts_per_user'],
                                                 shuffle=False)

    data_generator_valid = EriskDataGeneratorBow(user_level_data, subjects_split, set_type="valid",
                                                 hyperparams_features=hyperparams_features,
                                                 seq_len=hyperparams['maxlen'], batch_size=1,
                                                 max_posts_per_user=hyperparams['max_posts_per_user'],
                                                 shuffle=False)

    data_generator_test = EriskDataGeneratorBow(user_level_data, subjects_split, set_type="test",
                                                hyperparams_features=hyperparams_features,
                                                seq_len=hyperparams['maxlen'], batch_size=1,
                                                max_posts_per_user=hyperparams['max_posts_per_user'],
                                                shuffle=False)
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_daic(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DAICDataGenerator(user_level_data, subjects_split, set_type='train',
                                             hyperparams_features=hyperparams_features,
                                             seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                             max_posts_per_user=hyperparams['max_posts_per_user'],
                                             shuffle=False)

    data_generator_valid = DAICDataGenerator(user_level_data, subjects_split, set_type="valid",
                                             hyperparams_features=hyperparams_features,
                                             seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                             max_posts_per_user=hyperparams['max_posts_per_user'],
                                             shuffle=False)

    data_generator_test = DAICDataGenerator(user_level_data, subjects_split, set_type="test",
                                            hyperparams_features=hyperparams_features,
                                            seq_len=hyperparams['maxlen'], batch_size=1,
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            shuffle=False)
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_daic_raw(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DAICDataGeneratorRaw(user_level_data, subjects_split, set_type='train',
                                                hyperparams_features=hyperparams_features,
                                                seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                max_posts_per_user=hyperparams['max_posts_per_user'],
                                                shuffle=False)

    data_generator_valid = DAICDataGeneratorRaw(user_level_data, subjects_split, set_type="valid",
                                                hyperparams_features=hyperparams_features,
                                                seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                max_posts_per_user=hyperparams['max_posts_per_user'],
                                                shuffle=False)

    data_generator_test = DAICDataGeneratorRaw(user_level_data, subjects_split, set_type="test",
                                               hyperparams_features=hyperparams_features,
                                               seq_len=hyperparams['maxlen'], batch_size=1,
                                               max_posts_per_user=hyperparams['max_posts_per_user'],
                                               shuffle=False)
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_daic_bow(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DAICDataGeneratorBoW(user_level_data, subjects_split, set_type='train',
                                                hyperparams_features=hyperparams_features,
                                                seq_len=hyperparams['maxlen'], batch_size=1,
                                                max_posts_per_user=hyperparams['max_posts_per_user'],
                                                shuffle=False)

    data_generator_valid = DAICDataGeneratorBoW(user_level_data, subjects_split, set_type="valid",
                                                hyperparams_features=hyperparams_features,
                                                seq_len=hyperparams['maxlen'], batch_size=1,
                                                max_posts_per_user=hyperparams['max_posts_per_user'],
                                                shuffle=False)

    data_generator_test = DAICDataGeneratorBoW(user_level_data, subjects_split, set_type="test",
                                               hyperparams_features=hyperparams_features,
                                               seq_len=hyperparams['maxlen'], batch_size=1,
                                               max_posts_per_user=hyperparams['max_posts_per_user'],
                                               shuffle=False)
    return data_generator_train, data_generator_valid, data_generator_test

import json

from loader.EriskDataGenerator import EriskDataGenerator
from loader.EriskDataGenerator_raw import EriskDataGeneratorRaw
from loader.DAICDataGenerator_features import DAICDataGenerator
from loader.DAICDataGenerator_raw import DAICDataGeneratorRaw
from loader.DAICDataGenerator_bow import DAICDataGeneratorBoW


def initialize_datasets_erisk(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = EriskDataGenerator(user_level_data, subjects_split, set_type='train',
                                              hyperparams_features=hyperparams_features,
                                              seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                              max_posts_per_user=hyperparams['max_posts_per_user'],
                                              compute_liwc=True,
                                              shuffle=False,
                                              ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                              ablate_liwc='liwc' in hyperparams['ignore_layer'])

    data_generator_valid = EriskDataGenerator(user_level_data, subjects_split, set_type="valid",
                                              hyperparams_features=hyperparams_features,
                                              seq_len=hyperparams['maxlen'], batch_size=1,
                                              max_posts_per_user=hyperparams['max_posts_per_user'],
                                              shuffle=False,
                                              compute_liwc=True,
                                              ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                              ablate_liwc='liwc' in hyperparams['ignore_layer'])

    data_generator_test = EriskDataGenerator(user_level_data, subjects_split, set_type="test",
                                             hyperparams_features=hyperparams_features,
                                             seq_len=hyperparams['maxlen'], batch_size=1,
                                             max_posts_per_user=hyperparams['max_posts_per_user'],
                                             shuffle=False,
                                             compute_liwc=True,
                                             ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                             ablate_liwc='liwc' in hyperparams['ignore_layer'])
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_erisk_raw(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = EriskDataGeneratorRaw(user_level_data, subjects_split, set_type='train',
                                                 hyperparams_features=hyperparams_features,
                                                 seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                 max_posts_per_user=hyperparams['max_posts_per_user'],
                                                 compute_liwc=True,
                                                 shuffle=False,
                                                 ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                                 ablate_liwc='liwc' in hyperparams['ignore_layer'])

    data_generator_valid = EriskDataGeneratorRaw(user_level_data, subjects_split, set_type="valid",
                                                 hyperparams_features=hyperparams_features,
                                                 seq_len=hyperparams['maxlen'], batch_size=1,
                                                 max_posts_per_user=hyperparams['max_posts_per_user'],
                                                 shuffle=False,
                                                 compute_liwc=True,
                                                 ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                                 ablate_liwc='liwc' in hyperparams['ignore_layer'])

    data_generator_test = EriskDataGeneratorRaw(user_level_data, subjects_split, set_type="test",
                                                hyperparams_features=hyperparams_features,
                                                seq_len=hyperparams['maxlen'], batch_size=1,
                                                max_posts_per_user=hyperparams['max_posts_per_user'],
                                                shuffle=False,
                                                compute_liwc=True,
                                                ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                                ablate_liwc='liwc' in hyperparams['ignore_layer'])
    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_daic(
        hyperparams,
        hyperparams_features,
        path_train="../data/daic-woz/train_data.json",
        path_valid="../data/daic-woz/dev_data.json",
        path_test="../data/daic-woz/test_data.json",

):
    with open(path_train, "r") as f:
        data_json = json.load(f)

    data_generator_train = DAICDataGenerator(hyperparams_features=hyperparams_features,
                                             seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                             max_posts_per_user=hyperparams['max_posts_per_user'],
                                             post_groups_per_user=None,
                                             shuffle=True,
                                             compute_liwc=True,
                                             keep_first_batches=False)

    for session in data_json:
        data_generator_train.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    data_generator_train.generate_indexes()

    with open(path_valid, "r") as f:
        data_json = json.load(f)

    data_generator_valid = DAICDataGenerator(hyperparams_features=hyperparams_features,
                                             seq_len=hyperparams['maxlen'], batch_size=1,
                                             max_posts_per_user=hyperparams['max_posts_per_user'],
                                             post_groups_per_user=None,
                                             shuffle=False,
                                             compute_liwc=True,
                                             keep_first_batches=False)
    for session in data_json:
        data_generator_valid.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    data_generator_valid.generate_indexes()

    with open(path_test, "r") as f:
        data_json = json.load(f)

    data_generator_test = DAICDataGenerator(hyperparams_features=hyperparams_features,
                                            seq_len=hyperparams['maxlen'], batch_size=1,
                                            max_posts_per_user=hyperparams['max_posts_per_user'],
                                            post_groups_per_user=None,
                                            shuffle=False,
                                            compute_liwc=True,
                                            keep_first_batches=False)
    for session in data_json:
        data_generator_test.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    data_generator_test.generate_indexes()

    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_daic_raw(
        hyperparams,
        hyperparams_features,
        path_train="../data/daic-woz/train_data.json",
        path_valid="../data/daic-woz/dev_data.json",
        path_test="../data/daic-woz/test_data.json",

):
    with open(path_train, "r") as f:
        data_json = json.load(f)

    data_generator_train = DAICDataGeneratorRaw(batch_size=hyperparams['batch_size'],
                                                max_posts_per_user=hyperparams['max_posts_per_user'],
                                                post_groups_per_user=None,
                                                shuffle=True,
                                                keep_first_batches=False)

    for session in data_json:
        data_generator_train.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    data_generator_train.generate_indexes()

    with open(path_valid, "r") as f:
        data_json = json.load(f)

    data_generator_valid = DAICDataGeneratorRaw(batch_size=1,
                                                max_posts_per_user=hyperparams['max_posts_per_user'],
                                                post_groups_per_user=None,
                                                shuffle=False,
                                                keep_first_batches=False)
    for session in data_json:
        data_generator_valid.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    data_generator_valid.generate_indexes()

    with open(path_test, "r") as f:
        data_json = json.load(f)

    data_generator_test = DAICDataGeneratorRaw(batch_size=1,
                                               max_posts_per_user=hyperparams['max_posts_per_user'],
                                               post_groups_per_user=None,
                                               shuffle=False,
                                               keep_first_batches=False)
    for session in data_json:
        data_generator_test.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    data_generator_test.generate_indexes()

    return data_generator_train, data_generator_valid, data_generator_test


def initialize_datasets_daic_bow(
        hyperparams,
        hyperparams_features,
        path_train="../data/daic-woz/train_data.json",
        path_valid="../data/daic-woz/dev_data.json",
        path_test="../data/daic-woz/test_data.json",

):
    with open(path_train, "r") as f:
        data_json = json.load(f)

    data_generator_train = DAICDataGeneratorBoW(shuffle=True)

    for session in data_json:
        data_generator_train.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    vectorizer = data_generator_train.generate_indexes()

    with open(path_valid, "r") as f:
        data_json = json.load(f)

    data_generator_valid = DAICDataGeneratorBoW(shuffle=False, vectorizer=vectorizer)
    for session in data_json:
        data_generator_valid.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    data_generator_valid.generate_indexes()

    with open(path_test, "r") as f:
        data_json = json.load(f)

    data_generator_test = DAICDataGeneratorBoW(shuffle=False, vectorizer=vectorizer)
    for session in data_json:
        data_generator_test.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    data_generator_test.generate_indexes()

    return data_generator_train, data_generator_valid, data_generator_test

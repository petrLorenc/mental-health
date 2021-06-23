import json

from loader.DataGenerator import DataGenerator
from loader.DAICDataGenerator_v2 import DAICDataGenerator_v2


def initialize_datasets_erisk(user_level_data, subjects_split, hyperparams, hyperparams_features):
    data_generator_train = DataGenerator(user_level_data, subjects_split, set_type='train',
                                         hyperparams_features=hyperparams_features,
                                         seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                         posts_per_group=hyperparams['posts_per_group'],
                                         post_groups_per_user=hyperparams['post_groups_per_user'],
                                         max_posts_per_user=hyperparams['posts_per_user'],
                                         compute_liwc=True,
                                         ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                         ablate_liwc='liwc' in hyperparams['ignore_layer'])

    data_generator_valid = DataGenerator(user_level_data, subjects_split, set_type="valid",
                                         hyperparams_features=hyperparams_features,
                                         seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                         posts_per_group=hyperparams['posts_per_group'],
                                         post_groups_per_user=1,
                                         max_posts_per_user=None,
                                         shuffle=False,
                                         compute_liwc=True,
                                         ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                         ablate_liwc='liwc' in hyperparams['ignore_layer'])

    data_generator_test = DataGenerator(user_level_data, subjects_split, set_type="test",
                                        hyperparams_features=hyperparams_features,
                                        seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                        posts_per_group=hyperparams['posts_per_group'],
                                        post_groups_per_user=1,
                                        max_posts_per_user=None,
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

    data_generator_train = DAICDataGenerator_v2(hyperparams_features=hyperparams_features,
                                                seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                max_posts_per_user=None,
                                                posts_per_group=hyperparams['posts_per_group'],
                                                post_groups_per_user=None,
                                                shuffle=True,
                                                compute_liwc=True,
                                                keep_first_batches=False)

    for session in data_json:
        data_generator_train.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    data_generator_train.generate_indexes()

    with open(path_valid, "r") as f:
        data_json = json.load(f)

    data_generator_valid = DAICDataGenerator_v2(hyperparams_features=hyperparams_features,
                                                seq_len=hyperparams['maxlen'], batch_size=1,
                                                max_posts_per_user=None,
                                                posts_per_group=hyperparams['posts_per_group'],
                                                post_groups_per_user=None,
                                                shuffle=False,
                                                compute_liwc=True,
                                                keep_first_batches=False)
    for session in data_json:
        data_generator_valid.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    data_generator_valid.generate_indexes()

    with open(path_test, "r") as f:
        data_json = json.load(f)

    data_generator_test = DAICDataGenerator_v2(hyperparams_features=hyperparams_features,
                                               seq_len=hyperparams['maxlen'], batch_size=1,
                                               max_posts_per_user=None,
                                               posts_per_group=hyperparams['posts_per_group'],
                                               post_groups_per_user=None,
                                               shuffle=False,
                                               compute_liwc=True,
                                               keep_first_batches=False)
    for session in data_json:
        data_generator_test.load_daic_data(session, nickname=session["label"]["Participant_ID"])
    data_generator_test.generate_indexes()

    return data_generator_train, data_generator_valid, data_generator_test

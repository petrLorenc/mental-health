import os
import sys
import json
import pickle
import logging

print(os.getcwd())

from comet_ml import Experiment, Optimizer
from tensorflow.keras import callbacks
from callbacks import FreezeLayer, WeightsHistory, LRHistory

from model import build_hierarchical_model
# from load_save_model import save_model_and_params
from loader.DataGenerator import DataGenerator
from loader.DAICDataGenerator import DAICDataGenerator
from loader.DAICDataGenerator_v2 import DAICDataGenerator_v2
from loader.data_loading import load_erisk_data
from resource_loading import load_NRC, load_LIWC, load_stopwords

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # When cudnn implementation not found, run this
os.environ[
    "CUDA_VISIBLE_DEVICES"] = "0"  # Note: when starting kernel, for gpu_available to be true, this needs to be run
# only reserve 1 GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def train_model(model, hyperparams,
                data_generator_train, data_generator_valid,
                epochs, class_weight, start_epoch=0, workers=1,
                callback_list=[], logger=None,

                model_path='/tmp/model',
                validation_set='valid',
                verbose=1):
    if not logger:
        logger = logging.getLogger('training')
        ch = logging.StreamHandler(sys.stdout)
        # create formatter
        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(ch)
        logger.setLevel(logging.DEBUG)

    logger.info("Initializing callbacks...\n")
    # Initialize callbacks
    freeze_layer = FreezeLayer(patience=hyperparams['freeze_patience'], set_to=not hyperparams['trainable_embeddings'])
    weights_history = WeightsHistory()

    lr_history = LRHistory()

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=hyperparams['reduce_lr_factor'],
                                            patience=hyperparams['reduce_lr_patience'], min_lr=0.000001, verbose=1)
    lr_schedule = callbacks.LearningRateScheduler(lambda epoch, lr:
                                                  lr if (epoch + 1) % hyperparams['scheduled_reduce_lr_freq'] != 0 else
                                                  lr * hyperparams['scheduled_reduce_lr_factor'], verbose=1)
    callbacks_dict = {'freeze_layer': freeze_layer, 'weights_history': weights_history,
                      'lr_history': lr_history,
                      'reduce_lr_plateau': reduce_lr,
                      'lr_schedule': lr_schedule}

    logging.info('Train...')

    history = model.fit_generator(data_generator_train,
                                  # steps_per_epoch=100,
                                  epochs=epochs, initial_epoch=start_epoch,
                                  class_weight=class_weight,
                                  validation_data=data_generator_valid,
                                  verbose=verbose,
                                  #               validation_split=0.3,
                                  workers=workers,
                                  use_multiprocessing=False,
                                  # max_queue_size=100,

                                  callbacks=[
                                                # callbacks.ModelCheckpoint(filepath='%s_best.h5' % model_path,
                                                # verbose=1, save_best_only=True, save_weights_only=True),
                                                # callbacks.EarlyStopping(patience=hyperparams[
                                                # 'early_stopping_patience'], restore_best_weights=True)
                                            ] + [
                                                callbacks_dict[c] for c in [
                                          # 'weights_history',
                                      ]])
    return model, history


def get_network_type(hyperparams):
    if 'lstm' in hyperparams['ignore_layer']:
        network_type = 'cnn'
    else:
        network_type = 'lstm'
    if 'user_encoded' in hyperparams['ignore_layer']:
        if 'bert_layer' not in hyperparams['ignore_layer']:
            network_type = 'bert'
        else:
            network_type = 'extfeatures'
    if hyperparams['hierarchical']:
        hierarch_type = 'hierarchical'
    else:
        hierarch_type = 'seq'
    return network_type, hierarch_type


def initialize_experiment(hyperparams, nrc_lexicon_path, emotions, pretrained_embeddings_path,
                          dataset_type, transfer_type, hyperparams_features):
    # experiment = Experiment(api_key="eoBdVyznAhfg3bK9pZ58ZSXfv",
    #                         project_name="mental", workspace="ananana", disabled=False)
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="6XP0ix9zkGMuM24VbrnVRHSbf",
        project_name="general",
        workspace="petr-lorenc",
        disabled=False
    )

    experiment.log_parameters(hyperparams_features)

    experiment.log_parameter('emotion_lexicon', nrc_lexicon_path)
    experiment.log_parameter('emotions', emotions)
    experiment.log_parameter('embeddings_path', pretrained_embeddings_path)
    experiment.log_parameter('dataset_type', dataset_type)
    experiment.log_parameter('transfer_type', transfer_type)
    experiment.add_tag(dataset_type)
    experiment.log_parameters(hyperparams)
    network_type, hierarch_type = get_network_type(hyperparams)
    experiment.add_tag(network_type)
    experiment.add_tag(hierarch_type)

    return experiment


def initialize_datasets(user_level_data, subjects_split, hyperparams, hyperparams_features,
                        validation_set, session=None):
    data_generator_train = DataGenerator(user_level_data, subjects_split, set_type='train',
                                         hyperparams_features=hyperparams_features,
                                         seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                         posts_per_group=hyperparams['posts_per_group'],
                                         post_groups_per_user=hyperparams['post_groups_per_user'],
                                         max_posts_per_user=hyperparams['posts_per_user'],
                                         compute_liwc=True,
                                         ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                         ablate_liwc='liwc' in hyperparams['ignore_layer'])
    data_generator_valid = DataGenerator(user_level_data, subjects_split, set_type=validation_set,
                                         hyperparams_features=hyperparams_features,
                                         seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                         posts_per_group=hyperparams['posts_per_group'],
                                         post_groups_per_user=1,
                                         max_posts_per_user=None,
                                         shuffle=False,
                                         compute_liwc=True,
                                         ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                         ablate_liwc='liwc' in hyperparams['ignore_layer'])

    return data_generator_train, data_generator_valid


def initialize_model(hyperparams, hyperparams_features,
                     logger=None, session=None, transfer=False):
    if not logger:
        logger = logging.getLogger('training')
        ch = logging.StreamHandler(sys.stdout)
        # create formatter
        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(ch)
        logger.setLevel(logging.DEBUG)
    logger.info("Initializing model...\n")
    if 'emotions' in hyperparams['ignore_layer']:
        emotions_dim = 0
    else:
        emotions = load_NRC(hyperparams_features['nrc_lexicon_path'])
        emotions_dim = len(emotions)
    if 'liwc' in hyperparams['ignore_layer']:
        liwc_categories_dim = 0
    else:
        liwc_categories = load_LIWC(hyperparams_features['liwc_path'])
        liwc_categories_dim = len(liwc_categories)
    if 'stopwords' in hyperparams['ignore_layer']:
        stopwords_dim = 0
    else:
        stopwords_list = load_stopwords(hyperparams_features['stopwords_path'])
        stopwords_dim = len(stopwords_list)

    # Initialize model
    model = build_hierarchical_model(hyperparams, hyperparams_features,
                                     emotions_dim, stopwords_dim, liwc_categories_dim,
                                     ignore_layer=hyperparams['ignore_layer'])

    # model.summary()
    return model


def train(data_generator_train, data_generator_valid,
          hyperparams, hyperparams_features,
          experiment, dataset_type, transfer_type, logger=None,
          version=0, epochs=50, start_epoch=0,
          session=None, model=None, transfer_layer=False):
    network_type, hierarch_type = get_network_type(hyperparams)
    for feature in ['LIWC', 'emotions', 'numerical_dense_layer', 'sparse_feat_dense_layer', 'user_encoded']:
        if feature in hyperparams['ignore_layer']:
            network_type += "no%s" % feature
    if not transfer_layer:
        model_path = 'models/%s_%s_%s%d' % (network_type, dataset_type, hierarch_type, version)
    else:
        model_path = 'models/%s_%s_%s_transfer_%s%d' % (
            network_type, dataset_type, hierarch_type, transfer_type, version)

    if not model:
        if transfer_layer:
            logger.info("Initializing pretrained model...\n")
        else:
            logger.info("Initializing model...\n")
        model = initialize_model(hyperparams, hyperparams_features,
                                 session=session, transfer=transfer_layer)
    model.summary()

    print(model_path)
    logger.info("Training model...\n")
    model, history = train_model(model, hyperparams,
                                 data_generator_train, data_generator_valid,
                                 epochs=epochs, start_epoch=start_epoch,
                                 class_weight={0: 1, 1: hyperparams['positive_class_weight']},
                                 callback_list=[
                                     'weights_history',
                                     'lr_history',
                                     'reduce_lr_plateau',
                                     'lr_schedule'
                                 ],
                                 model_path=model_path, workers=1,
                                 validation_set=validation_set)
    logger.info("Saving model...\n")
    try:
        save_model_and_params(model, model_path, hyperparams, hyperparams_features)
        experiment.log_parameter("model_path", model_path)
    except:
        logger.error("Could not save model.\n")

    return model, history


if __name__ == '__main__':
    dataset = "daic"

    logger = logging.getLogger('training')
    ch = logging.StreamHandler(sys.stdout)
    # create formatter
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    hyperparams = {"trainable_embeddings": True, "lstm_units": 128, "dense_bow_units": 20, "dense_sentence_units": 0,
                   "dense_numerical_units": 20, "filters": 100, "kernel_size": 5, "lstm_units_user": 32,
                   "dense_user_units": 0, "filters_user": 10, "kernel_size_user": 3, "bert_dense_units": 256,
                   "bert_finetune_layers": 0, "bert_trainable": False, "bert_pooling": "first", "transfer_units": 20,
                   "dropout": 0.1, "l2_dense": 0.00011, "l2_embeddings": 1e-07, "l2_bert": 0.0001, "norm_momentum": 0.1,
                   "ignore_layer": [], "decay": 0.001, "lr": 5e-05, "reduce_lr_factor": 0.5,
                   "reduce_lr_patience": 55, "scheduled_reduce_lr_freq": 95, "scheduled_reduce_lr_factor": 0.5,
                   "freeze_patience": 2000, "threshold": 0.5, "early_stopping_patience": 5, "positive_class_weight": 2,
                   "class_weights": {"0": 1304, "1": 1287, "2": 763}, "maxlen": 256, "posts_per_user": None,
                   "post_groups_per_user": None, "posts_per_group": 10, "batch_size": 32, "padding": "pre",
                   "hierarchical": True, "sample_seqs": False, "sampling_distr": "exp", "optimizer": "adam"}
    hyperparams_features = {
        "max_features": 20000,
        "embedding_dim": 100,
        "vocabulary_path": "../resources/generated/all_vocab_clpsych_erisk_stop_20000.pkl",
        "nrc_lexicon_path": "../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        "liwc_path": "../resources/liwc.dic",
        "stopwords_path": "../resources/stopwords.txt",
        "embeddings_path": "/media/petrlorenc/Data/glove/glove.840B.300d.txt",
        "liwc_words_cached": "../resources/generated/liwc_categories_for_vocabulary_erisk_clpsych_stop_20K.pkl"
    }

    pretrained_embeddings_path = hyperparams_features['embeddings_path']
    dataset_type = "depression"
    transfer_type = None
    validation_set = 'valid'

    nrc_lexicon_path = hyperparams_features["nrc_lexicon_path"]
    nrc_lexicon = load_NRC(nrc_lexicon_path)
    emotions = list(nrc_lexicon.keys())

    experiment = initialize_experiment(hyperparams=hyperparams, nrc_lexicon_path=nrc_lexicon_path, emotions=emotions,
                                       pretrained_embeddings_path=pretrained_embeddings_path, dataset_type=dataset_type,
                                       transfer_type=transfer_type, hyperparams_features=hyperparams_features)

    logger.info("Initializing datasets...\n")
    if dataset == "erisk":
        writings_df = pickle.load(open('data/writings_df_%s_liwc' % dataset_type, 'rb'))

        user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,
                                                                      hyperparams_features=hyperparams_features)

        data_generator_train, data_generator_valid = initialize_datasets(user_level_data, subjects_split,
                                                                         hyperparams, hyperparams_features,
                                                                         validation_set=validation_set)
    elif dataset == "daic":
        with open("../data/daic-woz/train_data.json", "r") as f:
            data_json = json.load(f)

        data_generator_train = DAICDataGenerator_v2(hyperparams_features=hyperparams_features,
                                                 seq_len=hyperparams['maxlen'], batch_size=16,
                                                 max_posts_per_user=None,
                                                 posts_per_group=hyperparams['posts_per_group'],
                                                 post_groups_per_user=None,
                                                 shuffle=False, return_subjects=True,
                                                 compute_liwc=True, chunk_level_datapoints=False,
                                                 keep_first_batches=True)

        for session in data_json:
            data_generator_train.load_daic_data(session, nickname=session["label"]["Participant_ID"])
        data_generator_train.generate_indexes()

        with open("../data/daic-woz/dev_data.json", "r") as f:
            data_json = json.load(f)

        data_generator_valid = DAICDataGenerator_v2(hyperparams_features=hyperparams_features,
                                                 seq_len=hyperparams['maxlen'], batch_size=16,
                                                 max_posts_per_user=None,
                                                 posts_per_group=hyperparams['posts_per_group'],
                                                 post_groups_per_user=None,
                                                 shuffle=False, return_subjects=True,
                                                 compute_liwc=True, chunk_level_datapoints=False,
                                                 keep_first_batches=True)
        for session in data_json:
            data_generator_valid.load_daic_data(session, nickname=session["label"]["Participant_ID"])
        data_generator_valid.generate_indexes()

    else:
        raise NotImplementedError(f"Dataset {dataset} not recognized")

    train(data_generator_train=data_generator_train, data_generator_valid=data_generator_valid, hyperparams=hyperparams,
          hyperparams_features=hyperparams_features, experiment=experiment,
          dataset_type=dataset_type, transfer_type=transfer_type, logger=logger)

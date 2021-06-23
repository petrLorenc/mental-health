from utils.logger import logger
from train_utils.experiment import get_network_type, initialize_experiment

import os
import pickle
import logging
import argparse

print(os.getcwd())

from tensorflow.keras import callbacks
from callbacks import FreezeLayer, WeightsHistory, LRHistory

from model import build_hierarchical_model
from load_save_model import save_model_and_params
from loader.data_loading import load_erisk_data
from resource_loading import load_NRC, load_LIWC, load_stopwords
from train_utils.dataset import initialize_datasets_daic, initialize_datasets_erisk

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# When cudnn implementation not found, run this
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Note: when starting kernel, for gpu_available to be true, this needs to be run only reserve 1 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def train_model(model, hyperparams,
                data_generator_train, data_generator_valid,
                epochs, class_weight, start_epoch=0, workers=1,
                callback_list=frozenset(),
                verbose=1):
    logger.info("Initializing callbacks...\n")
    # Initialize callbacks
    # freeze_layer = FreezeLayer(patience=hyperparams['freeze_patience'], set_to=not hyperparams['trainable_embeddings'])

    weights_history = WeightsHistory(experiment=experiment)
    lr_history = LRHistory(experiment=experiment)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=hyperparams['reduce_lr_factor'],
                                            patience=hyperparams['reduce_lr_patience'], min_lr=0.000001, verbose=1)
    lr_schedule = callbacks.LearningRateScheduler(lambda epoch, lr:
                                                  lr if (epoch + 1) % hyperparams['scheduled_reduce_lr_freq'] != 0 else
                                                  lr * hyperparams['scheduled_reduce_lr_factor'], verbose=1)
    callbacks_dict = {
        # 'freeze_layer': freeze_layer,
        'weights_history': weights_history,
        'lr_history': lr_history,
        'reduce_lr_plateau': reduce_lr,
        'lr_schedule': lr_schedule
    }

    logging.info('Train...')

    history = model.fit_generator(data_generator_train,
                                  # steps_per_epoch=100,
                                  epochs=epochs, initial_epoch=start_epoch,
                                  class_weight=class_weight,
                                  validation_data=data_generator_valid,
                                  verbose=verbose,
                                  workers=workers,
                                  use_multiprocessing=False,
                                  callbacks=[callbacks_dict[c] for c in callback_list])
    return model, history


def initialize_model(hyperparams, hyperparams_features, word_embedding_type="random", session=None, transfer=False):
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
                                     ignore_layer=hyperparams['ignore_layer'],
                                     word_embedding_type=word_embedding_type)

    # model.summary()
    return model


def train(data_generator_train, data_generator_valid,
          hyperparams, hyperparams_features,
          experiment, dataset_type, transfer_type,
          version=0, epochs=1, start_epoch=0,
          session=None, model=None, transfer_layer=False,
          word_embedding_type="random"):
    network_type, hierarchy_type = get_network_type(hyperparams)
    for feature in ['LIWC', 'emotions', 'numerical_dense_layer', 'sparse_feat_dense_layer', 'user_encoded']:
        if feature in hyperparams['ignore_layer']:
            network_type += "no%s" % feature
    if not transfer_layer:
        model_path = '../resources/models/%s_%s_%s%d' % (network_type, dataset_type, hierarchy_type, version)
    else:
        model_path = '../resources/models/%s_%s_%s_transfer_%s%d' % (
            network_type, dataset_type, hierarchy_type, transfer_type, version)

    if not model:
        if transfer_layer:
            logger.info("Initializing pretrained model...\n")
        else:
            logger.info("Initializing model without transfer layers...\n")
        model = initialize_model(hyperparams, hyperparams_features,
                                 session=session, transfer=transfer_layer,
                                 word_embedding_type=word_embedding_type)
    model.summary()

    print(model_path)
    logger.info("Training model...\n")
    model, history = train_model(model, hyperparams,
                                 data_generator_train, data_generator_valid,
                                 epochs=epochs, start_epoch=start_epoch,
                                 class_weight={0: 1, 1: hyperparams['positive_class_weight']},
                                 callback_list=frozenset([
                                     'weights_history',
                                     'lr_history',
                                     'reduce_lr_plateau',
                                     'lr_schedule'
                                 ]),
                                 workers=1)
    logger.info("Saving model...\n")
    try:
        save_model_and_params(model, model_path, hyperparams, hyperparams_features)
        experiment.log_parameter("model_path", model_path)
    except:
        logger.error("Could not save model.\n")

    res = model.evaluate(data_generator_test, batch_size=1, verbose=0)
    logger.debug(res)
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--version', metavar="-v", type=int, default=0, help='version of model')
    parser.add_argument('--dataset', metavar="-d", type=str, default="daic",
                        help='used dataset (supported "daic" or "erisk")')
    parser.add_argument('--embeddings', type=str, default="random",
                        help='used embeddings for words (supported "random" or "glove")')
    parser.add_argument('--epochs', metavar="-e", type=int, default=10, help='number of epochs')
    args = parser.parse_args()

    hyperparams = {"trainable_embeddings": True, "dense_bow_units": 20, "dense_sentence_units": 0,
                   "dense_numerical_units": 20, "filters": 100, "kernel_size": 5,
                   "dense_user_units": 0, "filters_user": 10, "kernel_size_user": 3, "transfer_units": 20,
                   "dropout": 0.1, "l2_dense": 0.00011, "l2_embeddings": 1e-07, "norm_momentum": 0.1,
                   "ignore_layer": [], "decay": 0.001, "lr": 5e-05, "reduce_lr_factor": 0.5,
                   "reduce_lr_patience": 55, "scheduled_reduce_lr_freq": 95, "scheduled_reduce_lr_factor": 0.5,
                   "freeze_patience": 2000, "threshold": 0.5, "early_stopping_patience": 5,

                   "positive_class_weight": 2,
                   "maxlen": 30,
                   "lstm_units": 64,
                   "lstm_units_user": 64,
                   "posts_per_group": 10,
                   "batch_size": 32,

                   "posts_per_user": None,
                   "post_groups_per_user": None, "padding": "pre",
                   "hierarchical": True, "optimizer": "adam"}
    hyperparams_features = {
        "max_features": 20000,
        "embedding_dim": 300,
        "vocabulary_path": "../resources/generated/all_vocab_clpsych_erisk_stop_20000.pkl",
        "nrc_lexicon_path": "../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        "liwc_path": "../resources/liwc.dic",
        "stopwords_path": "../resources/stopwords.txt",
        "embeddings_path": "../resources/embeddings/glove.840B.300d.txt",
        "liwc_words_cached": "../resources/generated/liwc_categories_for_vocabulary_erisk_clpsych_stop_20K.pkl"
    }

    pretrained_embeddings_path = hyperparams_features['embeddings_path']
    dataset_type = "depression"
    transfer_type = None
    dataset = args.dataset

    nrc_lexicon_path = hyperparams_features["nrc_lexicon_path"]
    nrc_lexicon = load_NRC(nrc_lexicon_path)
    emotions = list(nrc_lexicon.keys())

    experiment = initialize_experiment(hyperparams=hyperparams, nrc_lexicon_path=nrc_lexicon_path, emotions=emotions,
                                       pretrained_embeddings_path=pretrained_embeddings_path, dataset_type=dataset_type,
                                       transfer_type=transfer_type, hyperparams_features=hyperparams_features)

    logger.info("Initializing datasets...\n")
    if dataset == "erisk":
        writings_df = pickle.load(open('../data/eRisk/writings_df_%s_liwc' % dataset_type, 'rb'))

        user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,
                                                                      hyperparams_features=hyperparams_features)

        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_erisk(user_level_data,
                                                                                                    subjects_split,
                                                                                                    hyperparams,
                                                                                                    hyperparams_features)
    elif dataset == "daic":

        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_daic(hyperparams,
                                                                                                   hyperparams_features)

    else:
        raise NotImplementedError(f"Dataset {dataset} not recognized")

    train(data_generator_train=data_generator_train,
          data_generator_valid=data_generator_valid,
          hyperparams=hyperparams,
          hyperparams_features=hyperparams_features,
          experiment=experiment,
          dataset_type=dataset_type,
          transfer_type=transfer_type,
          version=args.version,
          epochs=args.epochs,
          word_embedding_type=args.embeddings)

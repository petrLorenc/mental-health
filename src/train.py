from utils.logger import logger
from train_utils.experiment import initialize_experiment

import os
import pickle
import argparse

print(os.getcwd())

import tensorflow as tf
import numpy as np
import random

tf.random.set_seed(43)
np.random.seed(43)
random.seed(43)

from tensorflow.keras import callbacks
from callbacks import FreezeLayer, WeightsHistory, LRHistory
from nltk.tokenize import RegexpTokenizer

from model.hierarchical_model import build_hierarchical_model
from model.lstm_str import build_lstm_with_str_input
from model.bow_logistic_regression import build_bow_log_regression_model
from model.lstm_vector import build_lstm_with_vector_input
from model.lstm_stateful import build_lstm_stateful_model

from utils_test import test, test_stateful

from load_save_model import save_model_and_params, load_params, load_saved_model_weights
from loader.data_loading import load_erisk_data, load_daic_data
from resource_loading import load_NRC, load_LIWC, load_list_from_file

from train_utils.dataset import initialize_datasets_hierarchical
from train_utils.dataset import initialize_datasets_str
from train_utils.dataset import initialize_datasets_bow
from train_utils.dataset import initialize_datasets_bigrams
from train_utils.dataset import initialize_datasets_use_vector
from train_utils.dataset import initialize_datasets_stateful

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

# When cudnn implementation not found, run this
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Note: when starting kernel, for gpu_available to be true, this needs to be run only reserve 1 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def train_model(model, hyperparams,
                data_generator_train, data_generator_valid,
                epochs, class_weight, start_epoch=0, workers=1,
                callback_list=frozenset(),
                verbose=1):
    logger.info("Initializing callbacks...\n")
    weights_history = WeightsHistory(experiment=experiment)
    lr_history = LRHistory(experiment=experiment)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=hyperparams['reduce_lr_factor'],
                                            patience=hyperparams['reduce_lr_patience'], min_lr=0.000001, verbose=1)
    lr_schedule = callbacks.LearningRateScheduler(lambda epoch, lr:
                                                  lr if (epoch + 1) % hyperparams['scheduled_reduce_lr_freq'] != 0 else
                                                  lr * hyperparams['scheduled_reduce_lr_factor'], verbose=1)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    model_checkpoint = callbacks.ModelCheckpoint(
        f'../resources/generated/{hyperparams["model"]}_{hyperparams["embeddings"]}_{hyperparams["version"]}_{hyperparams["note"]}_best_model.h5',
        monitor='val_loss', mode='min', save_best_only=True)
    callbacks_dict = {
        # 'freeze_layer': freeze_layer,
        'weights_history': weights_history,
        'lr_history': lr_history,
        'reduce_lr_plateau': reduce_lr,
        'lr_schedule': lr_schedule,
        'early_stopping': early_stopping,
        'model_checkpoint': model_checkpoint
    }

    logger.info("Training model...\n")
    if "stateful" in hyperparams["model"]:
        for e in range(epochs):
            for data, label, _ in data_generator_train.yield_data_grouped_by_users():
                model.reset_states()
                labels = np.tile([label], len(data)).astype(np.float32)
                for d, l in zip(data, labels):
                    model.train_on_batch(np.array(d).reshape((1, 1, hyperparams_features["embedding_dim"])), np.array(l).reshape(1, 1))
        history = None
    else:
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


def initialize_model(hyperparams, hyperparams_features):
    logger.info("Initializing models...\n")

    if hyperparams["model"].startswith("hierarchical"):
        emotions_dim = 0 if 'emotions' in hyperparams['ignore_layer'] else len(load_NRC(hyperparams_features['nrc_lexicon_path']))
        stopwords_dim = 0 if 'stopwords' in hyperparams['ignore_layer'] else len(load_list_from_file(hyperparams_features['stopwords_path']))

        liwc_categories_dim = 0
        if 'liwc' not in hyperparams['ignore_layer']:
            num2emo, whole_words, asterisk_words = load_LIWC(hyperparams_features['liwc_path'])
            liwc_categories_dim = len(num2emo)

        model = build_hierarchical_model(hyperparams, hyperparams_features,
                                         emotions_dim, stopwords_dim, liwc_categories_dim,
                                         ignore_layer=hyperparams['ignore_layer'],
                                         word_embedding_type=hyperparams["embeddings"])
    elif hyperparams["model"] == "lstm_str":
        model = build_lstm_with_str_input(hyperparams, hyperparams_features)

    elif hyperparams["model"] == "lstm_vector":
        model = build_lstm_with_vector_input(hyperparams, hyperparams_features)

    elif hyperparams["model"] == "log_regression":
        model = build_bow_log_regression_model(hyperparams, hyperparams_features)

    elif hyperparams["model"] == "lstm_stateful":
        model = build_lstm_stateful_model(hyperparams, hyperparams_features)

    else:
        raise NotImplementedError(f"Type of model {hyperparams['model']} is not supported yet")

    # model.summary()
    return model


def train(data_generator_train, data_generator_valid,
          hyperparams, hyperparams_features,
          experiment,
          start_epoch=0,
          model=None):
    model_path = f'../resources/models/{hyperparams["model"]}_{hyperparams["embeddings"]}_{hyperparams["version"]}_{hyperparams["note"]}'

    if not model:
        model = initialize_model(hyperparams, hyperparams_features)
    model.summary()

    print(model_path)
    model, history = train_model(model, hyperparams,
                                 data_generator_train, data_generator_valid,
                                 epochs=hyperparams["epochs"], start_epoch=start_epoch,
                                 class_weight={0: 1, 1: hyperparams['positive_class_weight']},
                                 callback_list=frozenset([
                                     'weights_history',
                                     'lr_history',
                                     'reduce_lr_plateau',
                                     'lr_schedule',
                                     'model_checkpoint',
                                     'early_stopping'
                                 ]),
                                 workers=1)
    logger.info("Saving model...\n")
    try:
        save_model_and_params(model, model_path, hyperparams, hyperparams_features)
        experiment.log_parameter("model_path", model_path)
    except:
        logger.error("Could not save model.\n")

    return model, history


if __name__ == '__main__':
    logger.info("Loading command line arguments...\n")


    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--version', metavar="-v", type=int, default=0, help='version of model')
    parser.add_argument('--dataset', metavar="-d", type=str, help='(supported "daic" or "erisk")')
    parser.add_argument('--embeddings', type=str, help='(supported "random", "glove" or "use")')
    parser.add_argument('--epochs', metavar="-e", type=int, help='number of epochs')
    parser.add_argument('--model', metavar="-t", type=str, help="(supported 'hierarchical', 'lstm' or 'bow')")
    parser.add_argument('--only_test', type=str2bool, help="Only test - loading trained model from disk")
    parser.add_argument('--smaller_data', type=str2bool, help="Only test data (small portion)")
    parser.add_argument('--note', type=str, default="default", help="Note")
    parser.add_argument('--vocabulary', type=str, help="BoW vocabulary name")
    args = parser.parse_args()
    logger.info(args)

    model_path = f'../resources/models/{args.model}_{args.embeddings}_{args.version}{"_" + args.note if args.note else ""}'
    if args.only_test:
        # load saved model
        hyperparams, hyperparams_features = load_params(model_path=model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        hyperparams, hyperparams_features = load_params("../resources/default_config")
        if args.model == "hierarchical":
            from model.hierarchical_model import hyperparams, hyperparams_features

            hyperparams_features["embedding_dim"] = 300
            if args.vocabulary is not None:
                hyperparams_features['vocabulary_path'] = os.path.join("../resources/generated", args.vocabulary)

        elif args.model == "hierarchicalRandom":
            from model.hierarchical_model import hyperparams, hyperparams_features

            hyperparams_features["embedding_dim"] = 30
            hyperparams["embeddings"] = "random"
            if args.vocabulary is not None:
                hyperparams_features['vocabulary_path'] = os.path.join("../resources/generated", args.vocabulary)

        elif args.model == "lstm_str":
            from model.lstm_str import hyperparams, hyperparams_features

            hyperparams_features["embedding_dim"] = 512

        elif args.model == "lstm_vector":
            from model.lstm_vector import hyperparams, hyperparams_features

            hyperparams_features["embedding_dim"] = 512

        elif args.model == "lstm_stateful":
            from model.lstm_stateful import hyperparams, hyperparams_features

            hyperparams_features["embedding_dim"] = 512

        elif args.model == "log_regression":
            from model.bow_logistic_regression import hyperparams, hyperparams_features

            hyperparams_features["vocabulary_path"] = os.path.join("../resources/generated", args.vocabulary)
            hyperparams_features["embedding_dim"] = "dynamic"
        else:
            raise Exception(f"Unknown model {args.model}")

    # override from command line
    hyperparams.update({k: v for k, v in vars(args).items() if v is not None})

    dataset = hyperparams["dataset"]

    logger.info("Initializing datasets...\n")
    if dataset == "erisk":
        writings_df = pickle.load(open('../data/eRisk/writings_df_depression_liwc', 'rb'))
        if args.smaller_data:
            writings_df = writings_df.sample(frac=0.1)

        user_level_data, subjects_split = load_erisk_data(writings_df)

    elif dataset == "daic":
        user_level_data, subjects_split = load_daic_data(path_train="../data/daic-woz/train_data.json",
                                                         path_valid="../data/daic-woz/dev_data.json",
                                                         path_test="../data/daic-woz/test_data.json",
                                                         include_only=["Participant"],
                                                         # include_only=["Ellie", "Participant"],
                                                         limit_size=args.smaller_data,
                                                         tokenizer=RegexpTokenizer(r'\w+'))
    else:
        raise NotImplementedError(f"Not recognized dataset {dataset}")

    if hyperparams["embeddings"] == "random" or hyperparams["embeddings"] == "glove":
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_hierarchical(user_level_data,
                                                                                                           subjects_split,
                                                                                                           hyperparams,
                                                                                                           hyperparams_features)
    elif hyperparams["embeddings"] == "use-str":
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_str(user_level_data,
                                                                                                  subjects_split,
                                                                                                  hyperparams,
                                                                                                  hyperparams_features)
    elif hyperparams["embeddings"] == "use-vector":
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_use_vector(user_level_data,
                                                                                                         subjects_split,
                                                                                                         hyperparams,
                                                                                                         hyperparams_features)
    elif hyperparams["embeddings"] == "use-stateful":
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_stateful(user_level_data,
                                                                                                       subjects_split,
                                                                                                       hyperparams,
                                                                                                       hyperparams_features)
    elif hyperparams["embeddings"] == "bow":
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_bow(user_level_data,
                                                                                                  subjects_split,
                                                                                                  hyperparams,
                                                                                                  hyperparams_features)
        if hyperparams_features["embedding_dim"] == "dynamic":
            hyperparams_features["embedding_dim"] = data_generator_train.get_input_dimension()
    elif hyperparams["embeddings"] == "bigrams":
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_bigrams(user_level_data,
                                                                                                      subjects_split,
                                                                                                      hyperparams,
                                                                                                      hyperparams_features)
        if hyperparams_features["embedding_dim"] == "dynamic":
            hyperparams_features["embedding_dim"] = data_generator_train.get_input_dimension()
    else:
        raise NotImplementedError(f"Embeddings {hyperparams['embeddings']} not implemented yet")

    experiment = initialize_experiment(hyperparams=hyperparams, hyperparams_features=hyperparams_features)

    if not hyperparams["only_test"]:
        model, history = train(data_generator_train=data_generator_train,
                               data_generator_valid=data_generator_valid,
                               hyperparams=hyperparams,
                               hyperparams_features=hyperparams_features,
                               experiment=experiment)
    else:
        model = load_saved_model_weights(model_path=model_path, hyperparams=hyperparams, hyperparams_features=hyperparams_features, h5=True)

    if hyperparams["model"] == "log_regression":
        from model.bow_logistic_regression import log_important_features

        vocabulary = {v: k for k, v in data_generator_valid.vectorizer.vocabulary_.items()}
        log_important_features(experiment=experiment, vocabulary=vocabulary, model=model)

    if "stateful" in hyperparams["model"]:
        test_stateful(model=model,
                      data_generator_valid=data_generator_valid,
                      data_generator_test=data_generator_test,
                      experiment=experiment,
                      hyperparams=hyperparams,
                      hyperparams_features=hyperparams_features)
    else:
        test(model=model,
             data_generator_valid=data_generator_valid,
             data_generator_test=data_generator_test,
             experiment=experiment,
             hyperparams=hyperparams)

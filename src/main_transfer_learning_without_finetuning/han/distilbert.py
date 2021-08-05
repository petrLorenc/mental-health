from utils.logger import logger
import comet_ml
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

from nltk.tokenize import RegexpTokenizer

from model.hierarchical_model import build_hierarchical_model
from test_utils.utils import test
from train_utils.utils import train

from loader.data_loading import load_erisk_data, load_daic_data
from utils.resource_loading import load_NRC, load_LIWC, load_list_from_file

from train_utils.dataset import initialize_datasets_hierarchical_precomputed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[1], enable=True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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


    embeddings = "distilbert.words.words"
    model = "han"
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--version', metavar="-v", type=int, default=0, help='version of model')
    parser.add_argument('--only_test', type=str2bool, help="Only test - loading trained model from disk", default=False)
    parser.add_argument('--smaller_data', type=str2bool, help="Only test data (small portion)", default=False)
    parser.add_argument('--note', type=str, default="transfer", help="Note")
    args = parser.parse_args()
    logger.info(args)

    model_path = f'../resources/models/{model}_{embeddings}_{args.version}_{args.note}'

    logger.info("Initializing datasets...\n")
    writings_df = pickle.load(open('../../../data/eRisk/writings_df_depression_liwc', 'rb'))

    user_level_data_source, subjects_split_source = load_erisk_data(writings_df)

    user_level_data_target, subjects_split_target = load_daic_data(path_train="../../../data/daic-woz/train_data.json",
                                                                   path_valid="../../../data/daic-woz/dev_data.json",
                                                                   path_test="../../../data/daic-woz/test_data.json",
                                                                   include_only=["Participant"],
                                                                   # include_only=["Ellie", "Participant"],
                                                                   limit_size=args.smaller_data,
                                                                   tokenizer=RegexpTokenizer(r'\w+'))

    hyperparams = {
        "embeddings": embeddings,
        "trainable_embeddings": False,
        "model": model,
        "dataset": args.dataset,
        "version": args.version,
        "note": args.note,

        "positive_class_weight": 1,  #
        "chunk_size": 10,  #
        "batch_size": 32,  #
        "max_seq_len": 50,  #
        "epochs": 100,

        "threshold": 0.5,
        "l2_dense": 0.00011,
        "l2_embeddings": 1e-07,
        "norm_momentum": 0.1,
        "lstm_units_user": 100,
        "decay": 0.001,

        "reduce_lr_factor": 0.9,
        "reduce_lr_patience": 1,
        "scheduled_reduce_lr_freq": 1,
        "scheduled_reduce_lr_factor": 0.9,
        "optimizer": "adam",
        "learning_rate": 0.005,
        "early_stopping_patience": 5,

        "dropout_rate": 0.1,  #
        "lstm_units": 256,  #
        "dense_bow_units": 10,  #
        "dense_numerical_units": 10,  #
        "dense_user_units": 10  #
    }
    hyperparams_features = {
        "embeddings_name": embeddings,
        "embedding_dim": 768,
        "precomputed_vectors_path": f"../../../data/{args.dataset}/precomputed_features/",
        "nrc_lexicon_path": "../../../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        "stopwords_path": "../../../resources/stopwords.txt",
        "liwc_path": "../../../resources/liwc.dic"
    }

    hyperparams["positive_class_weight"] = experiment.get_parameter("positive_class_weight")
    hyperparams["chunk_size"] = experiment.get_parameter("chunk_size")
    hyperparams["max_seq_len"] = experiment.get_parameter("max_seq_len")
    hyperparams["dropout_rate"] = experiment.get_parameter("dropout_rate")
    hyperparams["lstm_units"] = experiment.get_parameter("lstm_units")
    hyperparams["dense_bow_units"] = experiment.get_parameter("dense_bow_units")
    hyperparams["dense_numerical_units"] = experiment.get_parameter("dense_numerical_units")
    hyperparams["dense_user_units"] = experiment.get_parameter("dense_user_units")
    hyperparams["batch_size"] = experiment.get_parameter("batch_size")
    hyperparams["learning_rate"] = experiment.get_parameter("learning_rate")

    emotions_dim = len(load_NRC(hyperparams_features['nrc_lexicon_path']))
    stopwords_dim = len(load_list_from_file(hyperparams_features['stopwords_path']))

    num2emo, whole_words, asterisk_words = load_LIWC(hyperparams_features['liwc_path'])
    liwc_categories_dim = len(num2emo)

    hyperparams["emotions_dim"] = emotions_dim
    hyperparams["stopwords_dim"] = stopwords_dim
    hyperparams["liwc_categories_dim"] = liwc_categories_dim

    hyperparams_features["precomputed_vectors_path"] = f"../../../data/eRisk/precomputed_features/"

    data_generator_train_source, data_generator_valid_source, data_generator_test_source = initialize_datasets_hierarchical_precomputed(
        user_level_data_source,
        subjects_split_source,
        hyperparams,
        hyperparams_features)

    hyperparams_features["precomputed_vectors_path"] = f"../../../data/daic-woz/precomputed_features/"

    data_generator_train_target, data_generator_valid_target, data_generator_test_target = initialize_datasets_hierarchical_precomputed(
        user_level_data_target,
        subjects_split_target,
        hyperparams,
        hyperparams_features)
    model = build_hierarchical_model(hyperparams, hyperparams_features,
                                     emotions_dim, stopwords_dim, liwc_categories_dim,
                                     word_embedding_type=hyperparams_features["embeddings_name"])

    experiment = initialize_experiment(hyperparams=hyperparams, hyperparams_features=hyperparams_features, project_name="transfer_learning_without_finetuning")

    model, history = train(data_generator_train=data_generator_train_source,
                           data_generator_valid=data_generator_valid_source,
                           hyperparams=hyperparams,
                           hyperparams_features=hyperparams_features,
                           experiment=experiment,
                           model=model, model_path=model_path)

    test(model=model,
         data_generator_train=data_generator_train_target,
         data_generator_valid=data_generator_valid_target,
         data_generator_test=data_generator_test_target,
         experiment=experiment,
         hyperparams=hyperparams)

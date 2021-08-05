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

from model.vector_precomputed import build_lstm_with_vector_input_precomputed
from test_utils.utils import test
from train_utils.utils import train

from loader.data_loading import load_erisk_data, load_daic_data

from train_utils.dataset import initialize_datasets_precomputed_vector_sequence

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


    embeddings = "bert.vstack"
    embeddings_dim = 768
    model = "lstm"
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--version', metavar="-v", type=int, default=0, help='version of model')
    parser.add_argument('--only_test', type=str2bool, help="Only test - loading trained model from disk", default=False)
    parser.add_argument('--smaller_data', type=str2bool, help="Only test data (small portion)", default=False)
    parser.add_argument('--note', type=str, default="transfer", help="Note")
    args = parser.parse_args()
    logger.info(args)

    model_path = f'../resources/models/{model}_{embeddings}_{embeddings_dim}_{args.version}_{args.note}'

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
        "embeddings_dim": embeddings_dim,
        "model": model,
        "dataset": "eRisk",
        "version": args.version,
        "note": args.note,

        "positive_class_weight": 1,  #
        "chunk_size": 10,  #
        "batch_size": 32,  #
        "epochs": 100,

        "threshold": 0.5,

        "reduce_lr_factor": 0.9,
        "reduce_lr_patience": 1,
        "scheduled_reduce_lr_freq": 1,
        "scheduled_reduce_lr_factor": 0.9,
        "optimizer": "adam",
        "learning_rate": 0.005,
        "early_stopping_patience": 5,

        "dropout_rate": 0.1,  #
        "lstm_units": 256  #
    }
    hyperparams_features = {"embeddings_name": embeddings, "embedding_dim": embeddings_dim,
                            "precomputed_vectors_path": f"../../../data/eRisk/precomputed_features/"}

    hyperparams["positive_class_weight"] = 3
    hyperparams["chunk_size"] = 70
    hyperparams["dropout_rate"] = 0.4
    hyperparams["lstm_units"] = 100
    hyperparams["batch_size"] = 8
    hyperparams["learning_rate"] = 0.0005

    data_generator_train_source, data_generator_valid_source, data_generator_test_source = initialize_datasets_precomputed_vector_sequence(user_level_data_source,
                                                                                                                      subjects_split_source,
                                                                                                                      hyperparams,
                                                                                                                      hyperparams_features)

    hyperparams_features = {"embeddings_name": embeddings, "embedding_dim": embeddings_dim,
                            "precomputed_vectors_path": f"../../../data/daic-woz/precomputed_features/"}
    data_generator_train_target, data_generator_valid_target, data_generator_test_target = initialize_datasets_precomputed_vector_sequence(user_level_data_target,
                                                                                                                      subjects_split_target,
                                                                                                                      hyperparams,
                                                                                                                      hyperparams_features)

    model = build_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features)

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


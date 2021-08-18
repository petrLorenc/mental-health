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
from utils.load_save_model import load_params, load_saved_model_weights


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

    embeddings = "distilbert.words.words"
    model_name = "han"
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--version', metavar="-v", type=int, default=0, help='version of model')
    parser.add_argument('--dataset', metavar="-d", type=str, help='(supported "daic" or "erisk")')
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    logger.info(args)

    logger.info("Initializing datasets...\n")
    if args.dataset == "eRisk":
        writings_df = pickle.load(open('../data/eRisk/writings_df_depression_liwc', 'rb'))

        user_level_data, subjects_split = load_erisk_data(writings_df)

    elif args.dataset == "daic-woz":
        user_level_data, subjects_split = load_daic_data(path_train="../../../data/daic-woz/train_data.json",
                                                         path_valid="../../../data/daic-woz/dev_data.json",
                                                         path_test="../../../data/daic-woz/test_data.json",
                                                         include_only=["Participant"],
                                                         # include_only=["Ellie", "Participant"],
                                                         limit_size=False,
                                                         tokenizer=RegexpTokenizer(r'\w+'))
    else:
        raise NotImplementedError(f"Not recognized dataset {args.dataset}")

    hyperparams, hyperparams_features = load_params(model_path=args.model_path)

    emotions_dim = len(load_NRC(hyperparams_features['nrc_lexicon_path']))
    stopwords_dim = len(load_list_from_file(hyperparams_features['stopwords_path']))

    num2emo, whole_words, asterisk_words = load_LIWC(hyperparams_features['liwc_path'])
    liwc_categories_dim = len(num2emo)

    hyperparams["emotions_dim"] = emotions_dim
    hyperparams["stopwords_dim"] = stopwords_dim
    hyperparams["liwc_categories_dim"] = liwc_categories_dim

    data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_hierarchical_precomputed(user_level_data,
                                                                                                                   subjects_split,
                                                                                                                   hyperparams,
                                                                                                                   hyperparams_features)

    model = build_hierarchical_model(hyperparams, hyperparams_features,
                                     emotions_dim, stopwords_dim, liwc_categories_dim,
                                     word_embedding_type=hyperparams_features["embeddings_name"])


    # model_path = f'../../../resources/models/{model_name}_{embeddings}_{args.version}_{str(random.random())}'
    print(args.model_path)

    print(hyperparams)
    print(hyperparams_features)

    model = load_saved_model_weights(loaded_model_structure=model, model_path=args.model_path, hyperparams=hyperparams,
                                     hyperparams_features=hyperparams_features, h5=True)

    experiment = initialize_experiment(hyperparams=hyperparams, hyperparams_features=hyperparams_features, project_name="daic-woz-optimization")
    experiment.log_parameters({"model_path": args.model_path})

    test(model=model,
         data_generator_train=data_generator_train,
         data_generator_valid=data_generator_valid,
         data_generator_test=data_generator_test,
         experiment=experiment,
         hyperparams=hyperparams)

    experiment.end()



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
from sklearn.model_selection import StratifiedKFold

from model.hierarchical_model import build_hierarchical_model
from test_utils.utils import test
from train_utils.utils import train
from utils.load_save_model import load_saved_model_weights, load_params

from loader.data_loading import load_erisk_data, load_daic_data
from utils.resource_loading import load_NRC, load_LIWC, load_list_from_file

from train_utils.dataset import initialize_datasets_hierarchical_precomputed

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# if gpus:
#     tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
#     tf.config.experimental.set_memory_growth(device=gpus[1], enable=True)
#
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class HyperparamSearch:

    def __init__(self, default_hyperparam, default_hyperparam_features, get_data_generator_fn, get_model_fn):
        self.default_hyperparam = default_hyperparam
        self.default_hyperparam_features = default_hyperparam_features

        self.get_data_generator_fn = get_data_generator_fn
        self.get_model_fn = get_model_fn

    def main(self):
        logger.info("Loading command line arguments...\n")

        parser = argparse.ArgumentParser(description='Train model')
        parser.add_argument('--version', metavar="-v", type=int, default=0, help='version of model')
        parser.add_argument('--note', type=str, default=None, help="Note")
        parser.add_argument('--model_path', type=str, help="Path to trained model")
        parser.add_argument('--gpu', type=int, default=0)
        args = parser.parse_args()
        logger.info(args)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        if gpus:
            tf.config.experimental.set_visible_devices(devices=gpus[args.gpu], device_type='GPU')
            tf.config.experimental.set_memory_growth(device=gpus[args.gpu], enable=True)

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        logger.info("Initializing datasets...\n")
        user_level_data_source, subjects_split_source = load_daic_data(path_train="/home/petlor/mental-health/code/data/daic-woz/train_data.json",
                                                                       path_valid="/home/petlor/mental-health/code/data/daic-woz/dev_data.json",
                                                                       path_test="/home/petlor/mental-health/code/data/daic-woz/test_data.json",
                                                                       include_only=["Participant"],
                                                                       # include_only=["Ellie", "Participant"],
                                                                       limit_size=False,
                                                                       tokenizer=RegexpTokenizer(r'\w+'))

        model_path = args.model_path
        hyperparams, hyperparams_features = load_params(model_path)
        hyperparams["dataset"] = "daic-woz"
        hyperparams_features["precomputed_vectors_path"] = hyperparams_features["precomputed_vectors_path"].replace("eRisk", "daic-woz")

        experiment = initialize_experiment(hyperparams=hyperparams, hyperparams_features=hyperparams_features,
                                           project_name="transfer-learning-optimization-wo")

        experiment.log_parameters(hyperparams)
        experiment.log_parameters(hyperparams_features)

        # model, history = train(data_generator_train=data_generator_source_train,
        #                        data_generator_valid=data_generator_source_valid,
        #                        hyperparams=hyperparams,
        #                        hyperparams_features=hyperparams_features,
        #                        experiment=experiment,
        #                        model=model, model_path=model_path)

        UAPs, UARs, uF1s = [], [], []

        data_generator_source_train, data_generator_source_valid, data_generator_source_test = self.get_data_generator_fn(
            user_level_data_source,
            subjects_split_source,
            hyperparams,
            hyperparams_features)

        model = self.get_model_fn(hyperparams, hyperparams_features)

        for _ in range(5):
            # tf.random.set_seed(43)
            # np.random.seed(43)
            # random.seed(43)
            model = load_saved_model_weights(model, model_path, hyperparams, hyperparams_features, h5=True)

            UAP, UAR, uF1 = test(model=model,
                                 data_generator_train=data_generator_source_train,    # TO ALLOW TO COMPARE RESULTS
                                 data_generator_valid=data_generator_source_train,    # TO ALLOW TO COMPARE RESULTS
                                 data_generator_test=data_generator_source_valid,     # TO ALLOW TO COMPARE RESULTS
                                 experiment=experiment,
                                 hyperparams=hyperparams)
            UAPs.append(UAP)
            UARs.append(UAR)
            uF1s.append(uF1)

        experiment.log_metric("daic_UAP", np.average(UAPs))
        experiment.log_metric("daic_UAR", np.average(UARs))
        experiment.log_metric("daic_uF1s", np.average(uF1s))

        print("daic_UAP: ", np.average(UAPs))
        print("daic_UAR: ", np.average(UARs))
        print("daic_uF1s: ", np.average(uF1s))

        experiment.end()

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
import json

tf.random.set_seed(43)
np.random.seed(43)
random.seed(43)

from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import StratifiedKFold

from model.hierarchical_model import build_hierarchical_model
from test_utils.utils import test
from train_utils.utils import train

from loader.data_loading import load_erisk_data, load_daic_data
from utils.resource_loading import load_NRC, load_LIWC, load_list_from_file

from train_utils.dataset import initialize_datasets_hierarchical_precomputed




class HyperparamSearch:

    def __init__(self, config, default_hyperparam, default_hyperparam_features, extract_from_experiment_fn, get_data_generator_fn, get_model_fn):
        self.config = config
        self.default_hyperparam = default_hyperparam
        self.default_hyperparam_features = default_hyperparam_features
        self.extract_from_experiment_fn = extract_from_experiment_fn
        self.get_data_generator_fn = get_data_generator_fn
        self.get_model_fn = get_model_fn

    def main(self):
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

        # embeddings = "distilbert.words.words"
        # model_name = "han"
        parser = argparse.ArgumentParser(description='Train model')
        parser.add_argument('--version', metavar="-v", type=int, default=0, help='version of model')
        parser.add_argument('--dataset', metavar="-d", type=str, help='(supported "daic" or "erisk")')
        parser.add_argument('--note', type=str, default=None, help="Note")
        parser.add_argument('--hp', type=str, default=None, help="Hyperparameters")
        parser.add_argument('--hpf', type=str, default=None, help="Hyperparameters features")
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
        if args.dataset == "eRisk":
            writings_df = pickle.load(open('/home/petlor/mental-health/code/data/eRisk/writings_df_depression_liwc', 'rb'))

            user_level_data, subjects_split = load_erisk_data(writings_df)

        elif args.dataset == "daic-woz":
            user_level_data, subjects_split = load_daic_data(path_train="/home/petlor/mental-health/code/data/daic-woz/train_data.json",
                                                             path_valid="/home/petlor/mental-health/code/data/daic-woz/dev_data.json",
                                                             path_test="/home/petlor/mental-health/code/data/daic-woz/test_data.json",
                                                             include_only=["Participant"],
                                                             # include_only=["Ellie", "Participant"],
                                                             limit_size=False,
                                                             tokenizer=RegexpTokenizer(r'\w+'))
        else:
            raise NotImplementedError(f"Not recognized dataset {args.dataset}")

        hyperparams = self.default_hyperparam
        hyperparams_features = self.default_hyperparam_features

        if args.hp and args.hpf:
            with open(args.hp, 'r') as hpf:
                hyperparams = json.load(hpf)
                print(hyperparams)
            with open(args.hpf, 'r') as hpff:
                hyperparams_features = json.load(hpff)
                hyperparams_features["precomputed_vectors_path"] = f"/home/petlor/mental-health/code/data/{args.dataset}/precomputed_features/"
                print(hyperparams_features)

        hyperparams["dataset"] = args.dataset
        hyperparams["version"] = args.version
        hyperparams["note"] = args.note

        if args.hp and args.hpf:
            UAPs, UARs, uF1s = [], [], []

            data_generator_train, data_generator_valid, data_generator_test = self.get_data_generator_fn(user_level_data,
                                                                                                         subjects_split,
                                                                                                         hyperparams,
                                                                                                         hyperparams_features)

            model = self.get_model_fn(hyperparams, hyperparams_features)

            random_id = str(random.random())
            model_path = f'/home/petlor/mental-health/code/resources/models/{hyperparams_features["model"]}_{hyperparams_features["embeddings_name"]}_{args.version}_{random_id}'

            experiment = initialize_experiment(hyperparams=hyperparams, hyperparams_features=hyperparams_features,
                                               project_name=f"{args.dataset}-optimization")
            experiment.log_parameters({f"model_path_{random_id}": model_path})
            logger.info(model_path)

            model, history = train(data_generator_train=data_generator_train,
                                   data_generator_valid=data_generator_valid,
                                   hyperparams=hyperparams,
                                   hyperparams_features=hyperparams_features,
                                   experiment=experiment,
                                   model=model, model_path=model_path)

            UAP, UAR, uF1 = test(model=model,
                                 data_generator_train=data_generator_train,
                                 data_generator_valid=data_generator_valid,
                                 data_generator_test=data_generator_test,
                                 experiment=experiment,
                                 hyperparams=hyperparams)

            UAPs.append(UAP)
            UARs.append(UAR)
            uF1s.append(uF1)

            experiment.log_metric("average_CV_UAP", np.average(UAPs))
            experiment.log_metric("average_CV_UAR", np.average(UARs))
            experiment.log_metric("average_CV_uF1s", np.average(uF1s))

            experiment.end()


        else:
            opt = comet_ml.Optimizer(self.config, api_key="6XP0ix9zkGMuM24VbrnVRHSbf")

            for experiment in opt.get_experiments(project_name=f"{args.dataset}-optimization"):
                self.extract_from_experiment_fn(experiment, hyperparams, hyperparams_features)

                experiment.log_parameters(hyperparams)
                experiment.log_parameters(hyperparams_features)

                UAPs, UARs, uF1s = [], [], []

                if args.dataset == "daic-woz":
                    skf = StratifiedKFold(n_splits=5)
                    for train_idx, valid_idx in skf.split(subjects_split["train"], [user_level_data[x]["label"] for x in subjects_split["train"]]):
                        cross_validation_subjects_split = {
                            # to be comparable with other paper
                            'train': [x for idx, x in enumerate(subjects_split["train"]) if idx in train_idx],
                            'valid': [x for idx, x in enumerate(subjects_split["train"]) if idx in valid_idx],
                            'test': subjects_split["valid"]
                        }

                        data_generator_train, data_generator_valid, data_generator_test = self.get_data_generator_fn(user_level_data,
                                                                                                                     cross_validation_subjects_split,
                                                                                                                     hyperparams,
                                                                                                                     hyperparams_features)

                        model = self.get_model_fn(hyperparams, hyperparams_features)

                        random_id = str(random.random())
                        model_path = f'/home/petlor/mental-health/code/resources/models/{hyperparams_features["model"]}_{hyperparams_features["embeddings_name"]}_{args.version}_{random_id}'
                        experiment.log_parameters({f"model_path_{random_id}": model_path})
                        logger.info(model_path)

                        model, history = train(data_generator_train=data_generator_train,
                                               data_generator_valid=data_generator_valid,
                                               hyperparams=hyperparams,
                                               hyperparams_features=hyperparams_features,
                                               experiment=experiment,
                                               model=model, model_path=model_path)

                        UAP, UAR, uF1 = test(model=model,
                                             data_generator_train=data_generator_train,
                                             data_generator_valid=data_generator_valid,
                                             data_generator_test=data_generator_test,
                                             experiment=experiment,
                                             hyperparams=hyperparams)

                        UAPs.append(UAP)
                        UARs.append(UAR)
                        uF1s.append(uF1)
                elif args.dataset == "eRisk":
                    data_generator_train, data_generator_valid, data_generator_test = self.get_data_generator_fn(user_level_data,
                                                                                                                 subjects_split,
                                                                                                                 hyperparams,
                                                                                                                 hyperparams_features)

                    model = self.get_model_fn(hyperparams, hyperparams_features)

                    random_id = str(random.random())
                    model_path = f'/home/petlor/mental-health/code/resources/models/{hyperparams_features["model"]}_{hyperparams_features["embeddings_name"]}_{args.version}_{random_id}'
                    experiment.log_parameters({f"model_path_{random_id}": model_path})
                    logger.info(model_path)

                    model, history = train(data_generator_train=data_generator_train,
                                           data_generator_valid=data_generator_valid,
                                           hyperparams=hyperparams,
                                           hyperparams_features=hyperparams_features,
                                           experiment=experiment,
                                           model=model, model_path=model_path)

                    UAP, UAR, uF1 = test(model=model,
                                         data_generator_train=data_generator_train,
                                         data_generator_valid=data_generator_valid,
                                         data_generator_test=data_generator_test,
                                         experiment=experiment,
                                         hyperparams=hyperparams)

                    UAPs.append(UAP)
                    UARs.append(UAR)
                    uF1s.append(uF1)

                experiment.log_metric("average_CV_UAP", np.average(UAPs))
                experiment.log_metric("average_CV_UAR", np.average(UARs))
                experiment.log_metric("average_CV_uF1s", np.average(uF1s))

                experiment.end()

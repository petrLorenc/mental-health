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

from nltk.tokenize import RegexpTokenizer

from model.hierarchical_model import build_hierarchical_model
from model.lstm_str_dan import build_lstm_with_str_input_dan
from model.lstm_str_tran import build_lstm_with_str_input_tran
from model.logistic_regression import build_logistic_regression_model
from model.lstm_vector_dan import build_lstm_with_vector_input_dan
from model.lstm_vector_tran import build_lstm_with_vector_input_tran
from model.lstm_stateful import build_lstm_stateful_model
from model.neural_network import build_neural_network_model
from model.neural_network_features import build_neural_network_model_features
from model.logistic_regression_features import build_logistic_regression_model_features
from model.vector_precomputed import build_lstm_with_vector_input_precomputed, build_attention_lstm_with_vector_input_precomputed, \
    build_attention_and_features_lstm_with_vector_input_precomputed, build_attention_and_aggregated_features_lstm_with_vector_input_precomputed

from test_utils.utils import test, test_stateful, test_attention
from train_utils.utils import train

from utils.load_save_model import load_params, load_saved_model_weights
from loader.data_loading import load_erisk_data, load_daic_data
from utils.resource_loading import load_NRC, load_LIWC, load_list_from_file

from train_utils.dataset import initialize_datasets_hierarchical
from train_utils.dataset import initialize_datasets_str
from train_utils.dataset import initialize_datasets_unigrams_features
from train_utils.dataset import initialize_datasets_unigrams
from train_utils.dataset import initialize_datasets_bigrams
from train_utils.dataset import initialize_datasets_tensorflowhub_vector
from train_utils.dataset import initialize_datasets_stateful
from train_utils.dataset import initialize_datasets_precomputed_vector_sequence
from train_utils.dataset import initialize_datasets_precomputed_vector_aggregated
from train_utils.dataset import initialize_datasets_precomputed_group_of_vectors_sequence

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[1], enable=True)

# When cudnn implementation not found, run this
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Note: when starting kernel, for gpu_available to be true, this needs to be run only reserve 1 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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


    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--version', metavar="-v", type=int, default=0, help='version of model')
    parser.add_argument('--dataset', metavar="-d", type=str, help='(supported "daic" or "erisk")', default="daic-woz")
    parser.add_argument('--embeddings', type=str, help='(supported "random", "glove" or "use")', default="use4.vstack")
    parser.add_argument('--embeddings_dim', type=int, default=512, help='(supported "random", "glove" or "use")', )
    parser.add_argument('--epochs', metavar="-e", type=int, help='number of epochs', default=50)
    parser.add_argument('--model', metavar="-t", type=str, help="(supported 'hierarchical', 'lstm' or 'bow')",
                        default="precomputed_embeddings_sequence_pure_lstm")
    parser.add_argument('--only_test', type=str2bool, help="Only test - loading trained model from disk", default=False)
    parser.add_argument('--smaller_data', type=str2bool, help="Only test data (small portion)", default=False)
    parser.add_argument('--note', type=str, default="default", help="Note")
    parser.add_argument('--vocabulary', type=str, help="BoW vocabulary name", default=None)
    parser.add_argument("--additional_feature_list", nargs="+", default=[])
    parser.add_argument("--additional_feature_aggregated", type=str2bool, default=False)

    args = parser.parse_args()
    logger.info(args)

    model_path = f'../resources/models/{args.model}_{args.embeddings}_{args.embeddings_dim}_{args.version}{"_" + args.note if args.note else ""}'

    logger.info("Initializing datasets...\n")
    if args.dataset == "eRisk":
        writings_df = pickle.load(open('../data/eRisk/writings_df_depression_liwc', 'rb'))
        if args.smaller_data:
            writings_df = writings_df.sample(frac=0.1)

        user_level_data, subjects_split = load_erisk_data(writings_df)

    elif args.dataset == "daic-woz":
        user_level_data, subjects_split = load_daic_data(path_train="../data/daic-woz/train_data.json",
                                                         path_valid="../data/daic-woz/dev_data.json",
                                                         path_test="../data/daic-woz/test_data.json",
                                                         include_only=["Participant"],
                                                         # include_only=["Ellie", "Participant"],
                                                         limit_size=args.smaller_data,
                                                         tokenizer=RegexpTokenizer(r'\w+'))
    else:
        raise NotImplementedError(f"Not recognized dataset {args.dataset}")

    hyperparams = {
        "positive_class_weight": 1,  #
        "chunk_size": 2,  #
        "batch_size": 64,  #
        "epochs": 50,

        "threshold": 0.5,

        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 55,
        "scheduled_reduce_lr_freq": 1,
        "scheduled_reduce_lr_factor": 0.5,
        "optimizer": "adam",
        "dropout_rate": 0,  #

        "dense_regularization_kernel": 0.01,  #
        "dense_regularization_activity": 0.01,  #

        "lstm_units": 128  #
    }

    hyperparams_features = {"embeddings_name": args.embeddings, "embedding_dim": args.embeddings_dim,
                            "precomputed_vectors_path": f"../data/{args.dataset}/precomputed_features/"}

    for positive_class_weight in [1, 2, 3, 5]:
        for chunk_size in [1, 2, 3, 5, 8, 10, 15, 25, 50]:
            for batch_size in [4, 8, 16, 32, 64]:
                for dropout_rate in [0, 0.1, 0.2, 0.5]:
                    for lstm_units in [16, 32, 64, 128, 256]:
                        hyperparams["positive_class_weight"] = positive_class_weight
                        hyperparams["chunk_size"] = chunk_size
                        hyperparams["batch_size"] = batch_size
                        hyperparams["dropout_rate"] = dropout_rate
                        hyperparams["lstm_units"] = lstm_units

                        # override from command line
                        hyperparams.update({k: v for k, v in vars(args).items() if v is not None})

                        experiment = initialize_experiment(hyperparams=hyperparams, hyperparams_features=hyperparams_features)
                        try:
                            logger.info("Experiment started")
                            if args.model.startswith("precomputed_embeddings_sequence"):
                                data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_precomputed_vector_sequence(
                                    user_level_data,
                                    subjects_split,
                                    hyperparams,
                                    hyperparams_features)
                                if args.model.endswith("attention_lstm") and len(args.additional_feature_list):
                                    additional_dimension = data_generator_train.additional_dimension
                                    if args.additional_feature_aggregated:
                                        model, attention_model = build_attention_and_aggregated_features_lstm_with_vector_input_precomputed(
                                            hyperparams, hyperparams_features,
                                            additional_features_dim=additional_dimension)
                                    else:
                                        model, attention_model = build_attention_and_features_lstm_with_vector_input_precomputed(hyperparams,
                                                                                                                                 hyperparams_features,
                                                                                                                                 additional_features_dim=additional_dimension)
                                elif args.model.endswith("attention_lstm"):
                                    model, attention_model = build_attention_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features)
                                elif args.model.endswith("pure_lstm"):
                                    model = build_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features)
                            elif args.model.startswith("precomputed_embeddings_aggregated"):
                                data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_precomputed_vector_aggregated(
                                    user_level_data,
                                    subjects_split,
                                    hyperparams,
                                    hyperparams_features)
                                if args.model.endswith("logistic_regression"):
                                    from model.logistic_regression import hyperparams

                                    model = build_logistic_regression_model(hyperparams, hyperparams_features)
                                elif args.model.endswith("neural_network"):
                                    from model.neural_network import hyperparams

                                    model = build_neural_network_model(hyperparams, hyperparams_features)
                                else:
                                    raise Exception("Unknown model {args.model}")
                            else:
                                raise Exception(f"Unknown model {args.model}")

                            if not hyperparams["only_test"]:
                                model, history = train(data_generator_train=data_generator_train,
                                                       data_generator_valid=data_generator_valid,
                                                       hyperparams=hyperparams,
                                                       hyperparams_features=hyperparams_features,
                                                       experiment=experiment,
                                                       model=model, model_path=model_path)
                            else:
                                model = load_saved_model_weights(loaded_model_structure=model, model_path=model_path, hyperparams=hyperparams,
                                                                 hyperparams_features=hyperparams_features, h5=True)

                            if hyperparams["model"] == "log_regression":
                                from model.logistic_regression import log_important_features

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
                                     data_generator_valid=data_generator_train,
                                     data_generator_test=data_generator_valid,
                                     experiment=experiment,
                                     hyperparams=hyperparams)

                                if "attention" in hyperparams["model"]:
                                    test_attention(attention_model=attention_model,
                                                   data_generator_valid=data_generator_valid,
                                                   data_generator_test=data_generator_test,
                                                   experiment=experiment,
                                                   hyperparams=hyperparams)
                        except Exception as e:
                            logger.info(str(e))
                            experiment.log_metric("error", True)
                        finally:
                            experiment.end()

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
from model.lstm_vector_distillbert import build_lstm_with_vector_input_precomputed

from test_utils.utils import test, test_stateful
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
    parser.add_argument('--dataset', metavar="-d", type=str, help='(supported "daic" or "erisk")')
    parser.add_argument('--embeddings', type=str, help='(supported "random", "glove" or "use")')
    parser.add_argument('--embeddings_dim', type=int, default=768, help='(supported "random", "glove" or "use")')
    parser.add_argument('--epochs', metavar="-e", type=int, help='number of epochs')
    parser.add_argument('--model', metavar="-t", type=str, help="(supported 'hierarchical', 'lstm' or 'bow')")
    parser.add_argument('--only_test', type=str2bool, help="Only test - loading trained model from disk")
    parser.add_argument('--smaller_data', type=str2bool, help="Only test data (small portion)")
    parser.add_argument('--note', type=str, default="default", help="Note")
    parser.add_argument('--vocabulary', type=str, help="BoW vocabulary name")
    args = parser.parse_args()
    logger.info(args)

    model_path = f'../resources/models/{args.dataset}_{args.model}_{args.version}{"_" + args.note if args.note else ""}'

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

    if args.model == "hierarchical" or args.model == "hierarchicalRandom":
        from model.hierarchical_model import hyperparams, hyperparams_features

        if args.only_test: hyperparams, hyperparams_features = load_params(model_path=model_path)  # rewrite param

        if args.model == "hierarchical":
            hyperparams_features["embedding_dim"] = 300
        else:
            hyperparams_features["embedding_dim"] = 30
            hyperparams["embeddings"] = "random"

        if args.vocabulary is not None:
            hyperparams_features['vocabulary_path'] = os.path.join("../resources/generated", args.vocabulary)

        emotions_dim = 0 if 'emotions' in hyperparams['ignore_layer'] else len(load_NRC(hyperparams_features['nrc_lexicon_path']))
        stopwords_dim = 0 if 'stopwords' in hyperparams['ignore_layer'] else len(load_list_from_file(hyperparams_features['stopwords_path']))

        liwc_categories_dim = 0
        if 'liwc' not in hyperparams['ignore_layer']:
            num2emo, whole_words, asterisk_words = load_LIWC(hyperparams_features['liwc_path'])
            liwc_categories_dim = len(num2emo)

        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_hierarchical(user_level_data,
                                                                                                           subjects_split,
                                                                                                           hyperparams,
                                                                                                           hyperparams_features)

        model = build_hierarchical_model(hyperparams, hyperparams_features,
                                         emotions_dim, stopwords_dim, liwc_categories_dim,
                                         ignore_layer=hyperparams['ignore_layer'],
                                         word_embedding_type=hyperparams["embeddings"])

    elif args.model == "lstm_str_dan" or args.model == "lstm_str_tran":
        if args.model == "lstm_str_dan":
            from model.lstm_str_dan import hyperparams, hyperparams_features
        else:
            from model.lstm_str_tran import hyperparams, hyperparams_features

        if args.only_test: hyperparams, hyperparams_features = load_params(model_path=model_path)  # rewrite param

        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_str(user_level_data,
                                                                                                  subjects_split,
                                                                                                  hyperparams,
                                                                                                  hyperparams_features)
        if args.model == "lstm_str_dan":
            model = build_lstm_with_str_input_dan(hyperparams, hyperparams_features)
        else:
            model = build_lstm_with_str_input_tran(hyperparams, hyperparams_features)

    elif args.model == "lstm_vector_dan" or args.model == "lstm_vector_tran":
        if args.model == "lstm_vector_dan":
            from model.lstm_vector_dan import hyperparams, hyperparams_features
        else:
            from model.lstm_vector_tran import hyperparams, hyperparams_features

        if args.only_test: hyperparams, hyperparams_features = load_params(model_path=model_path)  # rewrite param

        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_tensorflowhub_vector(user_level_data,
                                                                                                                   subjects_split,
                                                                                                                   hyperparams,
                                                                                                                   hyperparams_features)
        if args.model == "lstm_vector_dan":
            model = build_lstm_with_vector_input_dan(hyperparams, hyperparams_features)
        else:
            model = build_lstm_with_vector_input_tran(hyperparams, hyperparams_features)

    elif args.model == "lstm_stateful":
        from model.lstm_stateful import hyperparams, hyperparams_features

        if args.only_test: hyperparams, hyperparams_features = load_params(model_path=model_path)  # rewrite param

        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_stateful(user_level_data,
                                                                                                       subjects_split,
                                                                                                       hyperparams,
                                                                                                       hyperparams_features)

        model = build_lstm_stateful_model(hyperparams, hyperparams_features)

    elif args.model == "log_regression_unigrams" or args.model == "neural_network_unigrams":
        if args.model == "log_regression_unigrams":
            from model.logistic_regression import hyperparams, hyperparams_features
        else:
            from model.neural_network import hyperparams, hyperparams_features

        if args.only_test: hyperparams, hyperparams_features = load_params(model_path=model_path)  # rewrite param

        hyperparams_features["vocabulary_path"] = os.path.join("../resources/generated", args.vocabulary)
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_unigrams(user_level_data,
                                                                                                       subjects_split,
                                                                                                       hyperparams,
                                                                                                       hyperparams_features)
        hyperparams_features["embedding_dim"] = data_generator_train.get_input_dimension()

        if args.model == "log_regression_unigrams":
            model = build_logistic_regression_model(hyperparams, hyperparams_features)
        else:
            model = build_neural_network_model(hyperparams, hyperparams_features)

    elif args.model == "log_regression_bigrams" or args.model == "neural_network_bigrams":
        if args.model == "log_regression_bigrams":
            from model.neural_network import hyperparams, hyperparams_features
        else:
            from model.logistic_regression import hyperparams, hyperparams_features
        if args.only_test: hyperparams, hyperparams_features = load_params(model_path=model_path)  # rewrite param

        hyperparams_features["vocabulary_path"] = os.path.join("../resources/generated", args.vocabulary)
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_bigrams(user_level_data,
                                                                                                      subjects_split,
                                                                                                      hyperparams,
                                                                                                      hyperparams_features)
        hyperparams_features["embedding_dim"] = data_generator_train.get_input_dimension()

        if args.model == "log_regression_bigrams":
            model = build_logistic_regression_model(hyperparams, hyperparams_features)
        else:
            model = build_neural_network_model(hyperparams, hyperparams_features)

    elif args.model == "neural_network_unigrams_features":
        from model.neural_network import hyperparams, hyperparams_features

        if args.only_test: hyperparams, hyperparams_features = load_params(model_path=model_path)  # rewrite param

        hyperparams_features["vocabulary_path"] = os.path.join("../resources/generated", args.vocabulary)
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_unigrams_features(user_level_data,
                                                                                                                subjects_split,
                                                                                                                hyperparams,
                                                                                                                hyperparams_features)
        emotions_dim = len(load_NRC(hyperparams_features['nrc_lexicon_path']))

        num2emo, _, _ = load_LIWC(hyperparams_features['liwc_path'])
        liwc_categories_dim = len(num2emo)

        model = build_neural_network_model_features(hyperparams, hyperparams_features, emotions_dim, liwc_categories_dim)

    elif args.model.startswith("precomputed_embeddings_sequence"):
        from model.lstm_vector_distillbert import hyperparams

        hyperparams_features = {"embeddings_name": args.embeddings, "embedding_dim": args.embeddings_dim,
                                "precomputed_vectors_path": f"../data/{args.dataset}/precomputed_features/"}

        if args.only_test: hyperparams, hyperparams_features = load_params(model_path=model_path)  # rewrite param

        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_precomputed_vector_sequence(user_level_data,
                                                                                                                          subjects_split,
                                                                                                                          hyperparams,
                                                                                                                          hyperparams_features)
        if args.model.endswith("lstm"):
            model = build_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features)
    elif args.model.startswith("precomputed_embeddings_aggregated"):
        from model.lstm_vector_distillbert import hyperparams

        hyperparams_features = {"embeddings_name": args.embeddings, "embedding_dim": args.embeddings_dim,
                                "precomputed_vectors_path": f"../data/{args.dataset}/precomputed_features/"}

        if args.only_test: hyperparams, hyperparams_features = load_params(model_path=model_path)  # rewrite param

        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_precomputed_vector_aggregated(user_level_data,
                                                                                                                            subjects_split,
                                                                                                                            hyperparams,
                                                                                                                            hyperparams_features)
        if args.model.endswith("logistic_regression"):
            model = build_logistic_regression_model(hyperparams, hyperparams_features)
        elif args.model.endswith("neural_network"):
            model = build_neural_network_model(hyperparams, hyperparams_features)
        else:
            raise Exception("Unknown model {args.model}")
    else:
        raise Exception(f"Unknown model {args.model}")

    if not args.only_test:
        # override from command line
        hyperparams.update({k: v for k, v in vars(args).items() if v is not None})

    experiment = initialize_experiment(hyperparams=hyperparams, hyperparams_features=hyperparams_features)

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
             data_generator_valid=data_generator_valid,
             data_generator_test=data_generator_test,
             experiment=experiment,
             hyperparams=hyperparams)

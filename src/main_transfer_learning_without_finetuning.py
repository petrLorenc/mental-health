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
    parser.add_argument('--source_dataset', type=str, help='(supported "daic" or "erisk")')
    parser.add_argument('--target_dataset', type=str, help='(supported "daic" or "erisk")')
    parser.add_argument('--embeddings', type=str, help='(supported "random", "glove" or "use")')
    parser.add_argument('--embeddings_dim', type=int, help='(supported "random", "glove" or "use")')
    parser.add_argument('--epochs', metavar="-e", type=int, help='number of epochs')
    parser.add_argument('--model', metavar="-t", type=str, help="(supported 'hierarchical', 'lstm' or 'bow')")
    parser.add_argument('--note', type=str, default="default", help="Note")
    parser.add_argument('--vocabulary', type=str, help="BoW vocabulary name")
    parser.add_argument("--additional_feature_list", nargs="+", default=[])
    parser.add_argument("--additional_feature_aggregated", type=str2bool, default=False)


    args = parser.parse_args()
    logger.info(args)

    model_path = f'../resources/models/{args.model}_{args.embeddings}_{args.embeddings_dim}_{args.version}{"_" + args.note if args.note else ""}'

    logger.info("Initializing datasets...\n")
    if args.source_dataset == "eRisk":
        writings_df = pickle.load(open('../data/eRisk/writings_df_depression_liwc', 'rb'))

        all_data_source_domain, subjects_split_source_domain = load_erisk_data(writings_df)

    elif args.source_dataset == "daic-woz":
        all_data_source_domain, subjects_split_source_domain = load_daic_data(path_train="../data/daic-woz/train_data.json",
                                                                              path_valid="../data/daic-woz/dev_data.json",
                                                                              path_test="../data/daic-woz/test_data.json",
                                                                              include_only=["Participant"],
                                                                              # include_only=["Ellie", "Participant"],
                                                                              limit_size=False,
                                                                              tokenizer=RegexpTokenizer(r'\w+'))
    else:
        raise NotImplementedError(f"Not recognized source_dataset {args.source_dataset}")
    logger.info("Source dataset loaded")

    if args.target_dataset == "eRisk":
        writings_df = pickle.load(open('../data/eRisk/writings_df_depression_liwc', 'rb'))

        all_data_target_domain, subjects_split_target_domain = load_erisk_data(writings_df)

    elif args.target_dataset == "daic-woz":
        all_data_target_domain, subjects_split_target_domain = load_daic_data(path_train="../data/daic-woz/train_data.json",
                                                                              path_valid="../data/daic-woz/dev_data.json",
                                                                              path_test="../data/daic-woz/test_data.json",
                                                                              include_only=["Participant"],
                                                                              # include_only=["Ellie", "Participant"],
                                                                              limit_size=False,
                                                                              tokenizer=RegexpTokenizer(r'\w+'))
    else:
        raise NotImplementedError(f"Not recognized target_dataset {args.target_dataset}")
    logger.info("Target dataset loaded")

    if args.model == "hierarchical" or args.model == "hierarchicalRandom":
        from model.hierarchical_model import hyperparams, hyperparams_features

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

        data_generator_source_domain_train, data_generator_source_domain_valid, data_generator_source_domain_test = initialize_datasets_hierarchical(
            all_data_source_domain,
            subjects_split_source_domain,
            hyperparams,
            hyperparams_features)

        data_generator_target_domain_train, data_generator_target_domain_valid, data_generator_target_domain_test = initialize_datasets_hierarchical(
            all_data_target_domain,
            subjects_split_target_domain,
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

        data_generator_source_domain_train, data_generator_source_domain_valid, data_generator_source_domain_test = initialize_datasets_str(
            all_data_source_domain,
            subjects_split_source_domain,
            hyperparams,
            hyperparams_features)
        data_generator_target_domain_train, data_generator_target_domain_valid, data_generator_target_domain_test = initialize_datasets_str(
            all_data_target_domain,
            subjects_split_target_domain,
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

        data_generator_source_domain_train, data_generator_source_domain_valid, data_generator_source_domain_test = initialize_datasets_tensorflowhub_vector(
            all_data_source_domain,
            subjects_split_source_domain,
            hyperparams,
            hyperparams_features)
        data_generator_target_domain_train, data_generator_target_domain_valid, data_generator_target_domain_test = initialize_datasets_tensorflowhub_vector(
            all_data_target_domain,
            subjects_split_target_domain,
            hyperparams,
            hyperparams_features)
        if args.model == "lstm_vector_dan":
            model = build_lstm_with_vector_input_dan(hyperparams, hyperparams_features)
        else:
            model = build_lstm_with_vector_input_tran(hyperparams, hyperparams_features)

    elif args.model == "lstm_stateful":
        from model.lstm_stateful import hyperparams, hyperparams_features


        data_generator_source_domain_train, data_generator_source_domain_valid, data_generator_source_domain_test = initialize_datasets_stateful(
            all_data_source_domain,
            subjects_split_source_domain,
            hyperparams,
            hyperparams_features)
        data_generator_target_domain_train, data_generator_target_domain_valid, data_generator_target_domain_test = initialize_datasets_stateful(
            all_data_target_domain,
            subjects_split_target_domain,
            hyperparams,
            hyperparams_features)

        model = build_lstm_stateful_model(hyperparams, hyperparams_features)

    elif args.model == "log_regression_unigrams" or args.model == "neural_network_unigrams":
        if args.model == "log_regression_unigrams":
            from model.logistic_regression import hyperparams, hyperparams_features
        else:
            from model.neural_network import hyperparams, hyperparams_features


        hyperparams_features["vocabulary_path"] = os.path.join("../resources/generated", args.vocabulary)

        data_generator_source_domain_train, data_generator_source_domain_valid, data_generator_source_domain_test = initialize_datasets_unigrams(
            all_data_source_domain,
            subjects_split_source_domain,
            hyperparams,
            hyperparams_features)
        data_generator_target_domain_train, data_generator_target_domain_valid, data_generator_target_domain_test = initialize_datasets_unigrams(
            all_data_target_domain,
            subjects_split_target_domain,
            hyperparams,
            hyperparams_features)

        hyperparams_features["embedding_dim"] = data_generator_source_domain_train.get_input_dimension()

        if args.model == "log_regression_unigrams":
            model = build_logistic_regression_model(hyperparams, hyperparams_features)
        else:
            model = build_neural_network_model(hyperparams, hyperparams_features)

    elif args.model == "log_regression_bigrams" or args.model == "neural_network_bigrams":
        if args.model == "neural_network_bigrams":
            from model.neural_network import hyperparams, hyperparams_features
        else:
            from model.logistic_regression import hyperparams, hyperparams_features

        hyperparams_features["vocabulary_path"] = os.path.join("../resources/generated", args.vocabulary)
        data_generator_source_domain_train, data_generator_source_domain_valid, data_generator_source_domain_test = initialize_datasets_bigrams(
            all_data_source_domain,
            subjects_split_source_domain,
            hyperparams,
            hyperparams_features)
        data_generator_target_domain_train, data_generator_target_domain_valid, data_generator_target_domain_test = initialize_datasets_bigrams(
            all_data_target_domain,
            subjects_split_target_domain,
            hyperparams,
            hyperparams_features)
        hyperparams_features["embedding_dim"] = data_generator_source_domain_train.get_input_dimension()

        if args.model == "log_regression_bigrams":
            model = build_logistic_regression_model(hyperparams, hyperparams_features)
        else:
            model = build_neural_network_model(hyperparams, hyperparams_features)

    elif args.model == "neural_network_unigrams_features" or args.model == "log_regression_unigrams_features":
        from model.neural_network import hyperparams, hyperparams_features

        hyperparams_features["vocabulary_path"] = os.path.join("../resources/generated", args.vocabulary)
        data_generator_source_domain_train, data_generator_source_domain_valid, data_generator_source_domain_test = initialize_datasets_unigrams_features(
            all_data_source_domain,
            subjects_split_source_domain,
            hyperparams,
            hyperparams_features)
        data_generator_target_domain_train, data_generator_target_domain_valid, data_generator_target_domain_test = initialize_datasets_unigrams_features(
            all_data_target_domain,
            subjects_split_target_domain,
            hyperparams,
            hyperparams_features)
        emotions_dim = len(load_NRC(hyperparams_features['nrc_lexicon_path']))
        hyperparams_features["embedding_dim"] = data_generator_source_domain_train.get_input_dimension()

        num2emo, _, _ = load_LIWC(hyperparams_features['liwc_path'])
        liwc_categories_dim = len(num2emo)

        if args.model == "neural_network_unigrams_features":
            model = build_neural_network_model_features(hyperparams, hyperparams_features, emotions_dim, liwc_categories_dim)
        else:
            model = build_logistic_regression_model_features(hyperparams, hyperparams_features, emotions_dim, liwc_categories_dim)
    elif args.model.startswith("precomputed_embeddings_sequence"):
        from model.vector_precomputed import hyperparams

        hyperparams_features = {"embeddings_name": args.embeddings, "embedding_dim": args.embeddings_dim,
                                "precomputed_vectors_path": f"../data/{args.source_dataset}/precomputed_features/"}

        if len(args.additional_feature_list):
            data_generator_source_domain_train, data_generator_source_domain_valid, data_generator_source_domain_test = initialize_datasets_precomputed_group_of_vectors_sequence(
                all_data_source_domain,
                subjects_split_source_domain,
                hyperparams,
                hyperparams_features,
                args.additional_feature_list,
                args.additional_feature_aggregated)

            hyperparams_features["precomputed_vectors_path"] = f"../data/{args.target_dataset}/precomputed_features/"

            data_generator_target_domain_train, data_generator_target_domain_valid, data_generator_target_domain_test = initialize_datasets_precomputed_group_of_vectors_sequence(
                all_data_target_domain,
                subjects_split_target_domain,
                hyperparams,
                hyperparams_features,
                args.additional_feature_list,
                args.additional_feature_aggregated)
        else:
            data_generator_source_domain_train, data_generator_source_domain_valid, data_generator_source_domain_test = initialize_datasets_precomputed_vector_sequence(
                all_data_source_domain,
                subjects_split_source_domain,
                hyperparams,
                hyperparams_features)

            hyperparams_features["precomputed_vectors_path"] = f"../data/{args.target_dataset}/precomputed_features/"

            data_generator_target_domain_train, data_generator_target_domain_valid, data_generator_target_domain_test = initialize_datasets_precomputed_vector_sequence(
                all_data_target_domain,
                subjects_split_target_domain,
                hyperparams,
                hyperparams_features)

        if args.model.endswith("attention_lstm") and len(args.additional_feature_list):
            additional_dimension = data_generator_source_domain_train.additional_dimension
            if args.additional_feature_aggregated:
                model, attention_model = build_attention_and_aggregated_features_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features,
                                                                                                                    additional_features_dim=additional_dimension)
            else:
                model, attention_model = build_attention_and_features_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features,
                                                                                                         additional_features_dim=additional_dimension)
        elif args.model.endswith("attention_lstm"):
            model, attention_model = build_attention_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features)
        elif args.model.endswith("pure_lstm"):
            model = build_lstm_with_vector_input_precomputed(hyperparams, hyperparams_features)
    elif args.model.startswith("precomputed_embeddings_aggregated"):
        from utils.default_config import hyperparams

        hyperparams_features = {"embeddings_name": args.embeddings, "embedding_dim": args.embeddings_dim,
                                "precomputed_vectors_path": f"../data/{args.dataset}/precomputed_features/"}

        data_generator_source_domain_train, data_generator_source_domain_valid, data_generator_source_domain_test = initialize_datasets_precomputed_vector_aggregated(
            all_data_source_domain,
            subjects_split_source_domain,
            hyperparams,
            hyperparams_features)
        data_generator_target_domain_train, data_generator_target_domain_valid, data_generator_target_domain_test = initialize_datasets_precomputed_vector_aggregated(
            all_data_target_domain,
            subjects_split_target_domain,
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

    # override from command line
    hyperparams.update({k: v for k, v in vars(args).items() if v is not None})
    hyperparams["dataset"] = hyperparams["source_dataset"]
    hyperparams["only_test"] = False
    hyperparams["smaller_data"] = False

    experiment = initialize_experiment(hyperparams=hyperparams, hyperparams_features=hyperparams_features)

    model, history = train(data_generator_train=data_generator_source_domain_train,
                           data_generator_valid=data_generator_source_domain_valid,
                           hyperparams=hyperparams,
                           hyperparams_features=hyperparams_features,
                           experiment=experiment,
                           model=model, model_path=model_path)

    if "log_regression" in hyperparams["model"]:
        from model.logistic_regression import log_important_features

        vocabulary = {v: k for k, v in data_generator_source_domain_valid.vectorizer.vocabulary_.items()}
        log_important_features(experiment=experiment, vocabulary=vocabulary, model=model)

    if "stateful" in hyperparams["model"]:
        test_stateful(model=model,
                      data_generator_valid=data_generator_source_domain_valid,
                      data_generator_test=data_generator_source_domain_test,
                      experiment=experiment,
                      hyperparams=hyperparams,
                      hyperparams_features=hyperparams_features)
    else:
        test(model=model,
             data_generator_train=data_generator_target_domain_train,
             data_generator_valid=data_generator_target_domain_valid,
             data_generator_test=data_generator_target_domain_test,
             experiment=experiment,
             hyperparams=hyperparams)

        if "attention" in hyperparams["model"]:
            test_attention(attention_model=attention_model,
                           data_generator_valid=data_generator_source_domain_valid,
                           data_generator_test=data_generator_source_domain_test,
                           experiment=experiment,
                           hyperparams=hyperparams)

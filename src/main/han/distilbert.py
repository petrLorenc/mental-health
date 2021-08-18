from main.skeleton import HyperparamSearch

from train_utils.dataset import initialize_datasets_hierarchical_precomputed
from model.hierarchical_model import build_hierarchical_model
from utils.resource_loading import load_NRC, load_LIWC, load_list_from_file

embeddings = "distilbert.words.words"

# The optimization config:
config = {
    "algorithm": "bayes",
    "name": "Optimize HAN Network",
    "spec": {"maxCombo": 10, "objective": "maximize", "metric": "average_CV_UAR"},
    "parameters": {
        "lstm_units": {
            "type": "integer",
            "min": 32,
            "max": 256,
            "scalingType": "uniform",
        },
        "dense_bow_units": {
            "type": "integer",
            "min": 10,
            "max": 100,
            "scalingType": "uniform",
        },
        "dense_numerical_units": {
            "type": "integer",
            "min": 10,
            "max": 100,
            "scalingType": "uniform",
        },
        "dense_user_units": {
            "type": "integer",
            "min": 10,
            "max": 200,
            "scalingType": "uniform",
        },
        "chunk_size": {
            "type": "integer",
            "min": 1,
            "max": 80,
            "scalingType": "uniform",
        },
        "max_seq_len": {
            "type": "integer",
            "min": 10,
            "max": 100,
            "scalingType": "uniform",
        },
        "dropout_rate": {"type": "discrete", "values": [0.0, 0.1, 0.2, 0.3, 0.4]},
        "batch_size": {"type": "discrete", "values": [4, 8, 16, 32]},
        "positive_class_weight": {"type": "discrete", "values": [1, 2, 3]},
        "learning_rate": {"type": "discrete", "values": [0.1, 0.01, 0.001, 0.0005]},
    },
    "trials": 1,
}

hyperparams = {
    "trainable_embeddings": False,

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
    "nrc_lexicon_path": "../../../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "stopwords_path": "../../../resources/stopwords.txt",
    "liwc_path": "../../../resources/liwc.dic"
}

emotions_dim = len(load_NRC(hyperparams_features['nrc_lexicon_path']))
stopwords_dim = len(load_list_from_file(hyperparams_features['stopwords_path']))
num2emo, whole_words, asterisk_words = load_LIWC(hyperparams_features['liwc_path'])
liwc_categories_dim = len(num2emo)


def extract_from_experiment_fn(_experiment, _hyperparams, _hyperparams_features):
    _hyperparams["emotions_dim"] = emotions_dim
    _hyperparams["stopwords_dim"] = stopwords_dim
    _hyperparams["liwc_categories_dim"] = liwc_categories_dim
    _hyperparams_features["word_embedding_type"] = embeddings

    _hyperparams_features["precomputed_vectors_path"] = f"../../../data/{hyperparams['dataset']}/precomputed_features/"

    _hyperparams["positive_class_weight"] = _experiment.get_parameter("positive_class_weight")
    _hyperparams["chunk_size"] = _experiment.get_parameter("chunk_size")
    _hyperparams["max_seq_len"] = _experiment.get_parameter("max_seq_len")
    _hyperparams["dropout_rate"] = _experiment.get_parameter("dropout_rate")
    _hyperparams["lstm_units"] = _experiment.get_parameter("lstm_units")
    _hyperparams["dense_bow_units"] = _experiment.get_parameter("dense_bow_units")
    _hyperparams["dense_numerical_units"] = _experiment.get_parameter("dense_numerical_units")
    _hyperparams["dense_user_units"] = _experiment.get_parameter("dense_user_units")
    _hyperparams["batch_size"] = _experiment.get_parameter("batch_size")
    _hyperparams["learning_rate"] = _experiment.get_parameter("learning_rate")


if __name__ == '__main__':
    hps = HyperparamSearch(config=config,
                           default_hyperparam=hyperparams,
                           default_hyperparam_features=hyperparams_features,
                           get_model_fn=build_hierarchical_model,
                           get_data_generator_fn=initialize_datasets_hierarchical_precomputed,
                           extract_from_experiment_fn=extract_from_experiment_fn)
    hps.main()

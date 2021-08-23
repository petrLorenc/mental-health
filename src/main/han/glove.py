from main.skeleton import HyperparamSearch

from train_utils.dataset import initialize_datasets_hierarchical
from model.hierarchical_model import build_hierarchical_model
from utils.resource_loading import load_NRC, load_LIWC, load_list_from_file

embeddings = "glove"

# The optimization config:
config = {
    "algorithm": "bayes",
    "name": "Optimize HAN Network",
    "spec": {"maxCombo": 10, "objective": "maximize", "metric": "average_CV_UAR"},
    "parameters": {
        "lstm_units": {"type": "discrete", "values": [32, 44, 56, 68, 88, 106, 128, 186, 200]},
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
        "chunk_size": {"type": "discrete", "values": [2, 4, 6, 8, 10, 16, 22, 32, 64, 88, 128]},
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
    "trials": 2,
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
    "model": "han",
    "embeddings_name": embeddings,
    "embedding_dim": 300,
    "vocabulary_path": f"/home/petlor/mental-health/code/resources/generated/vocab_daic/unigrams_participant_3123.txt",
    "nrc_lexicon_path": "/home/petlor/mental-health/code/resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "stopwords_path": "/home/petlor/mental-health/code/resources/stopwords.txt",
    "liwc_path": "/home/petlor/mental-health/code/resources/liwc.dic",
    "embeddings_path": "/home/petlor/mental-health/code/resources/embeddings/glove.840B.300d.txt"
}

emotions_dim = len(load_NRC(hyperparams_features['nrc_lexicon_path']))
stopwords_dim = len(load_list_from_file(hyperparams_features['stopwords_path']))
num2emo, whole_words, asterisk_words = load_LIWC(hyperparams_features['liwc_path'])
liwc_categories_dim = len(num2emo)


def customize_hyperparams(hp, hpf):
    hp["emotions_dim"] = emotions_dim
    hp["stopwords_dim"] = stopwords_dim
    hp["liwc_categories_dim"] = liwc_categories_dim
    hpf["word_embedding_type"] = embeddings



def extract_from_experiment_fn(_experiment, _hyperparams, _hyperparams_features):
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
                           default_hyperparam=None,
                           default_hyperparam_features=None,
                           get_model_fn=build_hierarchical_model,
                           get_data_generator_fn=initialize_datasets_hierarchical,
                           extract_from_experiment_fn=extract_from_experiment_fn,
                           customize_hyperparams=customize_hyperparams)
    hps.main()

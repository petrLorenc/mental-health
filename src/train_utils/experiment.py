from utils.logger import logger
from comet_ml import Experiment, Optimizer


def get_network_type():
    network_type = 'lstm'
    hierarch_type = 'hierarchical'
    return network_type, hierarch_type


def initialize_experiment(hyperparams, nrc_lexicon_path, emotions, pretrained_embeddings_path, transfer_type, hyperparams_features):
    logger.info("Preparing Experiment (Comet_ml)...\n")
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="6XP0ix9zkGMuM24VbrnVRHSbf",
        project_name="general",
        workspace="petr-lorenc",
        disabled=False
    )

    experiment.log_parameters(hyperparams_features)

    experiment.log_parameter('emotion_lexicon', nrc_lexicon_path)
    experiment.log_parameter('emotions', emotions)
    experiment.log_parameter('embeddings_path', pretrained_embeddings_path)
    experiment.log_parameter('transfer_type', transfer_type)
    experiment.add_tag("depression")
    experiment.log_parameters(hyperparams)

    return experiment
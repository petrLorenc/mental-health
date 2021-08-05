from utils.logger import logger
from comet_ml import Experiment, Optimizer


def initialize_experiment(hyperparams, hyperparams_features, project_name):
    logger.info("Preparing Experiment (Comet_ml)...\n")
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="6XP0ix9zkGMuM24VbrnVRHSbf",
        project_name=project_name,
        workspace="petr-lorenc",
        disabled=False
    )

    experiment.add_tag(hyperparams["embeddings"])
    experiment.add_tag(hyperparams["dataset"])
    experiment.add_tag(hyperparams["model"])
    experiment.log_parameters(hyperparams)
    experiment.log_parameters(hyperparams_features)

    return experiment

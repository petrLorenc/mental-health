from utils.logger import logger

import numpy as np
import random
from tensorflow.keras import callbacks
from train_utils.callbacks import WeightsHistory, LRHistory

from utils.load_save_model import save_model_and_params


def train_model(model, hyperparams, hyperparams_features,
                data_generator_train, data_generator_valid,
                epochs, class_weight, experiment, start_epoch=0, workers=1,
                callback_list=frozenset(),
                verbose=1):
    logger.info("Initializing callbacks...\n")
    weights_history = WeightsHistory(experiment=experiment)
    lr_history = LRHistory(experiment=experiment)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=hyperparams['reduce_lr_factor'],
                                            patience=hyperparams['reduce_lr_patience'], min_lr=0.000001, verbose=1)
    lr_schedule = callbacks.LearningRateScheduler(lambda epoch, lr:
                                                  lr if (epoch + 1) % hyperparams['scheduled_reduce_lr_freq'] != 0 else
                                                  lr * hyperparams['scheduled_reduce_lr_factor'], verbose=1)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=hyperparams['early_stopping_patience'])
    model_checkpoint = callbacks.ModelCheckpoint(
        f'/home/petlor/mental-health/code/resources/generated/{hyperparams_features["model"]}_{hyperparams_features["embeddings_name"]}_{hyperparams["version"]}_{str(random.random())}_best_model.h5',
        monitor='val_loss', mode='min', save_best_only=True)
    callbacks_dict = {
        # 'freeze_layer': freeze_layer,
        'weights_history': weights_history,
        'lr_history': lr_history,
        'reduce_lr_plateau': reduce_lr,
        'lr_schedule': lr_schedule,
        'early_stopping': early_stopping,
        'model_checkpoint': model_checkpoint
    }

    logger.info("Training model...\n")
    if "stateful" in hyperparams_features["model"]:
        for e in range(epochs):
            for data, label, _ in data_generator_train.yield_data_grouped_by_users():
                model.reset_states()
                labels = np.tile([label], len(data)).astype(np.float32)
                for d, l in zip(data, labels):
                    model.train_on_batch(np.array(d).reshape((1, 1, hyperparams_features["embedding_dim"])), np.array(l).reshape(1, 1))
        history = None
    else:
        history = model.fit_generator(data_generator_train,
                                      # steps_per_epoch=100,
                                      epochs=epochs, initial_epoch=start_epoch,
                                      class_weight=class_weight,
                                      validation_data=data_generator_valid,
                                      verbose=verbose,
                                      workers=workers,
                                      use_multiprocessing=False,
                                      callbacks=[callbacks_dict[c] for c in callback_list])
    return model, history


def train(data_generator_train, data_generator_valid,
          hyperparams, hyperparams_features,
          experiment, model, model_path):
    model, history = train_model(model=model, hyperparams=hyperparams, hyperparams_features=hyperparams_features,
                                 data_generator_train=data_generator_train, data_generator_valid=data_generator_valid,
                                 epochs=hyperparams["epochs"], start_epoch=0,
                                 class_weight={0: 1, 1: hyperparams['positive_class_weight']},
                                 callback_list=frozenset([
                                     'weights_history',
                                     'lr_history',
                                     'reduce_lr_plateau',
                                     'lr_schedule',
                                     'model_checkpoint',
                                     'early_stopping'
                                 ]),
                                 workers=1, experiment=experiment)
    logger.info("Saving model...\n")
    # try:
    save_model_and_params(model, model_path, hyperparams, hyperparams_features)
    experiment.log_parameter("model_path", model_path)
    # except:
    #     logger.error("Could not save model.\n")

    return model, history


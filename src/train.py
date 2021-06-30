from utils.logger import logger
from train_utils.experiment import initialize_experiment

import os
import pickle
import argparse

print(os.getcwd())

from tensorflow.keras import callbacks
from callbacks import FreezeLayer, WeightsHistory, LRHistory
import tensorflow as tf
import numpy as np

from model.hierarchical_model import build_hierarchical_model
from model.lstm import build_lstm_model
from model.bow_logistic_regression import build_bow_log_regression_model

from load_save_model import save_model_and_params, load_params, load_saved_model_weights
from loader.data_loading import load_erisk_data, load_daic_data
from resource_loading import load_NRC, load_LIWC, load_list_from_file

from train_utils.dataset import initialize_datasets_hierarchical
from train_utils.dataset import initialize_datasets_raw
from train_utils.dataset import initialize_datasets_bow

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


def train_model(model, hyperparams,
                data_generator_train, data_generator_valid,
                epochs, class_weight, start_epoch=0, workers=1,
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
    callbacks_dict = {
        # 'freeze_layer': freeze_layer,
        'weights_history': weights_history,
        'lr_history': lr_history,
        'reduce_lr_plateau': reduce_lr,
        'lr_schedule': lr_schedule
    }

    logger.info("Training model...\n")
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


def initialize_model(hyperparams, hyperparams_features, word_embedding_type="random", model_type="hierarchical"):
    logger.info("Initializing models...\n")

    if model_type == "hierarchical":
        emotions_dim = 0 if 'emotions' in hyperparams['ignore_layer'] else len(load_NRC(hyperparams_features['nrc_lexicon_path']))
        liwc_categories_dim = 0 if 'liwc' in hyperparams['ignore_layer'] else len(load_LIWC(hyperparams_features['liwc_path']))
        stopwords_dim = 0 if 'stopwords' in hyperparams['ignore_layer'] else len(load_list_from_file(hyperparams_features['stopwords_path']))

        model = build_hierarchical_model(hyperparams, hyperparams_features,
                                         emotions_dim, stopwords_dim, liwc_categories_dim,
                                         ignore_layer=hyperparams['ignore_layer'],
                                         word_embedding_type=word_embedding_type)
    elif model_type == "lstm":
        model = build_lstm_model(hyperparams, hyperparams_features)

    elif model_type == "log_regression" and word_embedding_type == "bow":
        model = build_bow_log_regression_model(hyperparams, hyperparams_features)

    else:
        raise NotImplementedError(f"Type of model {model_type} is not supported yet")

    # model.summary()
    return model


def train(data_generator_train, data_generator_valid,
          hyperparams, hyperparams_features,
          experiment, args,
          start_epoch=0,
          model=None):
    model_path = f'../resources/models/{args.model}_{args.embeddings}_{args.version}_{args.note}'

    if not model:
        model = initialize_model(hyperparams, hyperparams_features,
                                 word_embedding_type=args.embeddings,
                                 model_type=args.model)
    model.summary()

    print(model_path)
    model, history = train_model(model, hyperparams,
                                 data_generator_train, data_generator_valid,
                                 epochs=args.epochs, start_epoch=start_epoch,
                                 class_weight={0: 1, 1: hyperparams['positive_class_weight']},
                                 callback_list=frozenset([
                                     'weights_history',
                                     'lr_history',
                                     'reduce_lr_plateau',
                                     'lr_schedule'
                                 ]),
                                 workers=1)
    logger.info("Saving model...\n")
    try:
        save_model_and_params(model, model_path, hyperparams, hyperparams_features)
        experiment.log_parameter("model_path", model_path)
    except:
        logger.error("Could not save model.\n")

    return model, history


def test(model, data_generator_valid, data_generator_test, experiment, logger, hyperparams):
    logger.info("Testing model...\n")

    ratios = []
    ground_truth = []

    step = 0
    for data, label, _ in data_generator_valid.yield_data_grouped_by_users():
        prediction = model.predict_on_batch(data)
        experiment.log_histogram_3d(prediction, "Confidences - validation set", step=step)
        # threshold for one sequence (typically set to 0.5)
        prediction_for_user = [x for x in map(lambda x: x > hyperparams["threshold"], prediction)]
        if len(prediction_for_user) > 1:
            # working with sequences
            ratio_of_depressed_sequences = sum(prediction_for_user) / len(prediction_for_user)
        else:
            # working with classification of whole datapoint for user (typically BoW)
            ratio_of_depressed_sequences = prediction[0]
        ratios.append(ratio_of_depressed_sequences)
        ground_truth.append(label)
        step += 1

    experiment.log_histogram_3d(values=ratios, name="valid_ratio")

    # find best threshold for ratio
    best_threshold = 0.0
    best_UAR = 0.5  # Unweighted Average Recall (UAR)
    for tmp_threshold in np.linspace(0, 1, 50):
        tmp_prediction = [int(x[0]) for x in map(lambda x: x > tmp_threshold, ratios)]
        tmp_tp = sum([t == 1 and t == p for t, p in zip(ground_truth, tmp_prediction)])
        tmp_tn = sum([t == 0 and t == p for t, p in zip(ground_truth, tmp_prediction)])
        tmp_fp = sum([t == 0 and p == 1 for t, p in zip(ground_truth, tmp_prediction)])
        tmp_fn = sum([t == 1 and p == 0 for t, p in zip(ground_truth, tmp_prediction)])

        tmp_recall_1 = float(tmp_tp) / (float(tmp_tp + tmp_fn) + tf.keras.backend.epsilon())
        tmp_recall_0 = float(tmp_tn) / (float(tmp_tn + tmp_fp) + tf.keras.backend.epsilon())
        tmp_UAR = (tmp_recall_1 + tmp_recall_0) / 2
        if tmp_UAR > best_UAR:
            best_UAR = tmp_UAR
            best_threshold = tmp_threshold

    data_identifications = []
    ratios = []
    ground_truth = []

    step = 0
    for data, label, data_identification in data_generator_test.yield_data_grouped_by_users():
        prediction = model.predict_on_batch(data)
        experiment.log_histogram_3d(prediction, "Confidences - test set", step=step)
        # threshold for one sequence (typically set to 0.5)
        prediction_for_user = [x for x in map(lambda x: x > hyperparams["threshold"], prediction)]
        if len(prediction_for_user) > 1:
            # working with sequences
            ratio_of_depressed_sequences = sum(prediction_for_user) / len(prediction_for_user)
        else:
            # working with classification of whole datapoint for user (typically BoW)
            ratio_of_depressed_sequences = prediction[0]

        ratios.append(ratio_of_depressed_sequences)
        ground_truth.append(label)
        data_identifications.append(data_identification)
        step += 1

    experiment.log_histogram_3d(values=ratios, name="test_ratio")

    predictions = [int(x[0]) for x in map(lambda x: x > best_threshold, ratios)]
    tp = sum([t == 1 and t == p for t, p in zip(ground_truth, predictions)])
    tn = sum([t == 0 and t == p for t, p in zip(ground_truth, predictions)])
    fp = sum([t == 0 and p == 1 for t, p in zip(ground_truth, predictions)])
    fn = sum([t == 1 and p == 0 for t, p in zip(ground_truth, predictions)])

    recall_1 = float(tp) / (float(tp + fn) + tf.keras.backend.epsilon())
    recall_0 = float(tn) / (float(tn + fp) + tf.keras.backend.epsilon())
    precision_0 = float(tp) / (float(tp + fp) + tf.keras.backend.epsilon())
    precision_1 = float(tn) / (float(tn + fn) + tf.keras.backend.epsilon())
    experiment.log_metric("test_recall_1", recall_1)
    experiment.log_metric("test_recall_0", recall_0)
    experiment.log_metric("test_precision_1", precision_0)
    experiment.log_metric("test_precision_0", precision_1)

    logger.debug(f"Recall_0: {recall_0}, Recall_1:{recall_1}, Precision_0:{precision_0}, Precision_1:{precision_1}")

    experiment.log_confusion_matrix(y_true=ground_truth,
                                    y_predicted=predictions,
                                    labels=["0", "1"],
                                    index_to_example_function=lambda x: data_identifications[x])


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
    parser.add_argument('--dataset', metavar="-d", type=str, default="daic", help='(supported "daic" or "erisk")')
    parser.add_argument('--embeddings', type=str, default="random", help='(supported "random", "glove" or "use")')
    parser.add_argument('--epochs', metavar="-e", type=int, default=10, help='number of epochs')
    parser.add_argument('--model', metavar="-t", type=str, default="hierarchical", help="(supported 'hierarchical', 'lstm' or 'bow')")
    parser.add_argument('--only_test', type=str2bool, help="Only test - loading trained model from disk")
    parser.add_argument('--smaller_data', type=str2bool, help="Only test data (small portion)")
    parser.add_argument('--note', type=str, help="Note")
    parser.add_argument('--bow_vocabulary', type=str, help="BoW vocabulary name")
    args = parser.parse_args()
    logger.info(args)

    model_path = f'../resources/models/{args.model}_{args.embeddings}_{args.version}_{args.note}'
    if args.only_test:
        # load saved model
        hyperparams, hyperparams_features = load_params(model_path=model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        hyperparams, hyperparams_features = load_params("../resources/default_config")
        if args.embeddings == "glove":
            hyperparams_features["embedding_dim"] = 300
        elif args.embeddings == "random":
            hyperparams_features["embedding_dim"] = 30
        elif args.embeddings == "use":
            hyperparams_features["embedding_dim"] = 512
        elif args.embeddings == "bow":
            hyperparams_features["bow_vocabulary"] = os.path.join("../resources/generated", args.bow_vocabulary)
            hyperparams_features["embedding_dim"] = "dynamic"

    dataset = args.dataset

    experiment = initialize_experiment(hyperparams=hyperparams, args=args, hyperparams_features=hyperparams_features)

    logger.info("Initializing datasets...\n")
    if dataset == "erisk":
        writings_df = pickle.load(open('../data/eRisk/writings_df_depression_liwc', 'rb'))
        if args.smaller_data:
            writings_df = writings_df.sample(frac=0.1)

        liwc_dict = load_LIWC(hyperparams_features['liwc_path'])
        liwc_categories = set(liwc_dict.keys())
        user_level_data, subjects_split = load_erisk_data(writings_df, liwc_categories=liwc_categories)

    elif dataset == "daic":
        user_level_data, subjects_split = load_daic_data(path_train="../data/daic-woz/train_data.json",
                                                         path_valid="../data/daic-woz/dev_data.json",
                                                         path_test="../data/daic-woz/test_data.json",
                                                         limit_size=args.smaller_data)
    else:
        raise NotImplementedError(f"Not recognized dataset {dataset}")

    if args.embeddings == "random" or args.embeddings == "glove":
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_hierarchical(user_level_data,
                                                                                                           subjects_split,
                                                                                                           hyperparams,
                                                                                                           hyperparams_features)
    elif args.embeddings == "use":
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_raw(user_level_data,
                                                                                                  subjects_split,
                                                                                                  hyperparams,
                                                                                                  hyperparams_features)
    elif args.embeddings == "bow":
        data_generator_train, data_generator_valid, data_generator_test = initialize_datasets_bow(user_level_data,
                                                                                                  subjects_split,
                                                                                                  hyperparams,
                                                                                                  hyperparams_features)
        if hyperparams_features["embedding_dim"] == "dynamic":
            hyperparams_features["embedding_dim"] = data_generator_train.get_input_dimension()
    else:
        raise NotImplementedError(f"Embeddings {args.embeddings} not implemented yet")

    if not args.only_test:
        model, history = train(data_generator_train=data_generator_train,
                               data_generator_valid=data_generator_valid,
                               hyperparams=hyperparams,
                               hyperparams_features=hyperparams_features,
                               experiment=experiment,
                               args=args)
    else:
        model = load_saved_model_weights(model_path=model_path, hyperparams=hyperparams, hyperparams_features=hyperparams_features, h5=True,
                                         args=args)

    test(model=model,
         data_generator_valid=data_generator_valid,
         data_generator_test=data_generator_test,
         experiment=experiment,
         logger=logger,
         hyperparams=hyperparams)

from utils.logger import logger

import numpy as np
import tensorflow as tf

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


def test(model, data_generator_valid, data_generator_test, experiment, hyperparams):
    logger.info("Testing model...\n")

    best_threshold = get_best_threshold_based_on_validation_set(model, experiment, data_generator_valid, hyperparams)

    data_identifications = []
    ratios = []
    ground_truth = []

    step = 0
    for gen_data, label, data_identification in data_generator_test.yield_data_grouped_by_users():
        prediction = []
        for d in gen_data:
            p = model.predict_on_batch(d)
            prediction.append(p)
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
    log_results(experiment, ground_truth, predictions, data_identifications)


def test_stateful(model, data_generator_valid, data_generator_test, experiment, hyperparams, hyperparams_features):
    logger.info("Testing model...\n")

    predictions, ground_truth, data_identifications = [], [], []

    step = 0
    for data, label, data_identification in data_generator_test.yield_data_grouped_by_users():
        prediction = []
        for one_timestamp in data:
            prediction.append(model.predict(one_timestamp.reshape(-1, 1, hyperparams_features["embedding_dim"])))
        model.reset_states()

        experiment.log_histogram_3d(prediction, "Confidences - test set", step=step)

        predictions.append(int(prediction[-1][0][0] > 0.1))
        ground_truth.append(label)
        data_identifications.append(data_identification)
        step += 1

    log_results(experiment, ground_truth, predictions, data_identifications)


def get_best_threshold_based_on_validation_set(model, experiment, data_generator_valid, hyperparams):
    ratios = []
    ground_truth = []

    step = 0
    for gen_data, label, _ in data_generator_valid.yield_data_grouped_by_users():
        prediction = []
        for d in gen_data:
            p = model.predict_on_batch(d)
            prediction.append(p)
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
    return best_threshold


def log_results(experiment, ground_truth, predictions, data_identifications):
    tp = sum([t == 1 and t == p for t, p in zip(ground_truth, predictions)])
    tn = sum([t == 0 and t == p for t, p in zip(ground_truth, predictions)])
    fp = sum([t == 0 and p == 1 for t, p in zip(ground_truth, predictions)])
    fn = sum([t == 1 and p == 0 for t, p in zip(ground_truth, predictions)])

    recall_1 = float(tp) / (float(tp + fn) + tf.keras.backend.epsilon())
    recall_0 = float(tn) / (float(tn + fp) + tf.keras.backend.epsilon())
    precision_1 = float(tp) / (float(tp + fp) + tf.keras.backend.epsilon())
    precision_0 = float(tn) / (float(tn + fn) + tf.keras.backend.epsilon())

    experiment.log_metric("test_recall_1", recall_1)
    experiment.log_metric("test_recall_0", recall_0)
    experiment.log_metric("test_precision_1", precision_1)
    experiment.log_metric("test_precision_0", precision_0)

    experiment.log_metric("test_UAR", float(recall_0 + recall_1) / 2.0)
    experiment.log_metric("test_UAP", float(precision_0 + precision_1) / 2.0)
    experiment.log_metric("test_F1", f1_score(y_true=ground_truth, y_pred=predictions, average='macro'))

    UAP, UAR, uF1, _ = precision_recall_fscore_support(y_true=ground_truth, y_pred=predictions, labels=[0, 1])

    logger.debug(f"Recall 0: {recall_0}, Recall 1:{recall_1}, Precision 0:{precision_0}, Precision 1:{precision_1}")
    logger.debug(f"(sklearn) Recall 0: {UAR[0]}, Recall 1:{UAR[1]}, Precision 0:{UAP[0]}, Precision 1:{UAP[1]}, F1 0: {uF1[0]}, F1 1: {uF1[1]}")

    logger.debug(f"UAR: {float(recall_0 + recall_1) / 2.0}")
    logger.debug(f"UAR (sklearn): {np.average(UAR)}")
    logger.debug(f"UAP: {float(precision_0 + precision_1) / 2.0}")
    logger.debug(f"UAP (sklearn): {np.average(UAP)}")
    logger.debug(f"F1: {np.average([(2*recall_0 * precision_0)/(recall_0 + precision_0 + tf.keras.backend.epsilon()), (2*recall_1 * precision_1)/(recall_1 + precision_1 + tf.keras.backend.epsilon())])}")
    logger.debug(f"F1 (sklearn): {f1_score(y_true=ground_truth, y_pred=predictions, average='macro')}")
    logger.debug(f"F1 (sklearn 2): {np.average(uF1)} for {uF1} (class 1, 0)")

    experiment.log_confusion_matrix(y_true=ground_truth,
                                    y_predicted=predictions,
                                    labels=["0", "1"],
                                    index_to_example_function=lambda x: data_identifications[x])

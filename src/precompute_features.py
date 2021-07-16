"""
Each file with vectors will have name of subject will be saved as pickle
"""
import os
import numpy as np
import pickle as plk
from nltk.tokenize import RegexpTokenizer

import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel

from loader.data_loading import load_erisk_data, load_daic_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[1], enable=True)

if __name__ == '__main__':
    processing_batch = 25
    dataset = "eRisk"
    feature_extraction = "distilbert-base-uncased"
    embedding_dim = 768
    dir_to_save = f"../data/{dataset}/precomputed_features/"

    print(f"Checking directory to save features {dir_to_save}")
    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)

    print("Loading feature extractor...")
    tokenizer = DistilBertTokenizer.from_pretrained(feature_extraction)
    vectorizer = TFDistilBertModel.from_pretrained(feature_extraction)

    print("Loading datasets...")
    if dataset == "eRisk":
        writings_df = plk.load(open('../data/eRisk/writings_df_depression_liwc', 'rb'))
        user_level_data, subjects_split = load_erisk_data(writings_df)

    elif dataset == "daic-woz":
        user_level_data, subjects_split = load_daic_data(path_train="../data/daic-woz/train_data.json",
                                                         path_valid="../data/daic-woz/dev_data.json",
                                                         path_test="../data/daic-woz/test_data.json",
                                                         include_only=["Participant"],
                                                         # include_only=["Ellie", "Participant"],
                                                         limit_size=False,
                                                         tokenizer=RegexpTokenizer(r'\w+'))
    else:
        raise Exception(f"Unknown dataset: {dataset}")

    for k in user_level_data.keys():
        raw_texts = user_level_data[k]["raw"]

        num_batches = len(raw_texts)//processing_batch
        preprocessed_vectors = []
        try:
            for idx in range(num_batches + 1):
                if len(raw_texts[idx * processing_batch: (idx + 1) * processing_batch]) > 0:
                    preprocessed_vectors.append(vectorizer(tokenizer(raw_texts[idx * processing_batch: (idx + 1) * processing_batch], return_tensors="tf", padding=True, truncation=True)).last_hidden_state.numpy()[:, 0])
            preprocessed_vectors = np.array([x for y in preprocessed_vectors for x in y], dtype=np.float32).reshape(-1, embedding_dim)
        except:
            print(f"{k} has errors with {len(raw_texts)}")
        with open(os.path.join(dir_to_save, k + f".feat.{feature_extraction}.plk"), "wb") as f:
            plk.dump(preprocessed_vectors, f)
        print(f"{k} preprocessed")




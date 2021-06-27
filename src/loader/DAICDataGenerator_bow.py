from utils.logger import logger

import re
import string
import random
import pickle
import numpy as np

from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import sequence
from resource_loading import load_NRC, load_LIWC, load_vocabulary, load_stopwords
from utils.feature_encoders import encode_emotions, encode_pronouns, encode_stopwords, encode_liwc_categories

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class DAICDataGeneratorBoW(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 vectorizer = None,
                 shuffle=True):
        'Initialization'
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.user_level_texts = {}
        self.shuffle = shuffle
        self.vectorizer = vectorizer
        self.frozen_users = []

    def load_daic_data(self, session, nickname=None):
        if nickname is not None:
            nickname = str(nickname)
        else:
            nickname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))

        if nickname not in self.user_level_texts.keys():
            self.user_level_texts[nickname] = {}
            self.user_level_texts[nickname]['texts'] = []
            self.user_level_texts[nickname]['raw'] = []
            self.user_level_texts[nickname]["label"] = []

        for datapoint in session["transcripts"]:
            words = []
            raw_text = ""
            # if datapoint["speaker"] == "Participant":
            if "value" in datapoint:
                tokenized_text = self.tokenizer.tokenize(datapoint["value"])
                words.extend(tokenized_text)
                raw_text += datapoint["value"]

                self.user_level_texts[nickname]['texts'].append(words)
                self.user_level_texts[nickname]['raw'].append(raw_text)
        self.user_level_texts[nickname]["label"].append(int(session["label"]["PHQ8_Binary"]))

    def generate_indexes(self):
        all_texts = [x for data in self.user_level_texts.values() for x in data["raw"]]
        self.vectorizer = CountVectorizer() if self.vectorizer is None else self.vectorizer
        self.vectorizer.fit(all_texts)
        self.on_epoch_end()
        return self.vectorizer

    def yield_data_for_user(self):
        for user, data in self.user_level_texts.items():
            label = self.user_level_texts[user]['label']
            user_texts = " ".join([x for x in data["raw"]])

            yield label, self.vectorizer.transform([user_texts]).toarray()

    def get_input_dimension(self):
        if self.vectorizer is not None:
            return len(self.vectorizer.get_feature_names())
        else:
            raise RuntimeError("Vectorizer is not set yet")

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.user_level_texts) - 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        user = self.frozen_users[index]
        X, y = self.yield_data_for_specific_user(user)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.frozen_users = list(self.user_level_texts.keys())
        if self.shuffle:
            np.random.shuffle(self.frozen_users)

    def yield_data_for_specific_user(self, user):
        """
        Generates data containing batch_size samples
        Args:
            indexes: ()

        Returns:

        """
        ''  # X : (n_samples, *dim, n_channels)
        label = self.user_level_texts[user]['label']
        user_texts = " ".join([x for x in self.user_level_texts[user]["raw"]])

        label = np.array(label, dtype=np.float32)

        return self.vectorizer.transform([user_texts]).toarray(), label

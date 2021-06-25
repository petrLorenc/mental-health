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


class DAICDataGeneratorRaw(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 batch_size,
                 post_groups_per_user=None, post_offset=0,
                 max_posts_per_user=10,
                 shuffle=True,
                 keep_last_batch=True,
                 keep_first_batches=False):
        'Initialization'
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.user_level_texts = {}
        self.batch_size = batch_size
        self.keep_last_batch = keep_last_batch
        self.shuffle = shuffle
        self.post_offset = post_offset
        self.max_posts_per_user = max_posts_per_user
        self.keep_first_batches = keep_first_batches  # in the rolling window case, whether it will keep

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
        self.indexes_per_user = {u: [] for u in self.user_level_texts.keys()}
        self.indexes_with_user = []
        for u, data in self.user_level_texts.items():
            user_posts = data['texts']

            # Rolling window of datapoints: chunks with overlapping posts
            nr_post_groups = len(user_posts)
            if self.keep_first_batches:
                # Generate datapoints for first posts, before a complete chunk
                for i in range(1, min(self.max_posts_per_user, nr_post_groups - 1)):
                    self.indexes_per_user[u].append(range(self.post_offset, i + self.post_offset))
                    self.indexes_with_user.append((u, range(self.post_offset, i + self.post_offset)))

            for i in range(nr_post_groups):
                # Stop at the last complete chunk
                if i + self.max_posts_per_user + self.post_offset > len(user_posts):
                    break
                self.indexes_per_user[u].append(range(i + self.post_offset, min(i + self.max_posts_per_user + self.post_offset, len(user_posts))))
                self.indexes_with_user.append((u, range(i + self.post_offset, min(i + self.max_posts_per_user + self.post_offset, len(user_posts)))))
        self.on_epoch_end()

    def yield_data_for_user(self):
        for user, index_array in self.indexes_per_user.items():
            label = self.user_level_texts[user]['label']
            user_texts = []
            for indexes in index_array:
                current_batch = np.array([self.user_level_texts[user]['raw'][i] for i in indexes])
                user_texts.append(current_batch)

            user_texts = np.array(user_texts, dtype=np.str).reshape(-1, self.max_posts_per_user)
            yield label, user_texts

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.keep_last_batch:
            return int(np.ceil(len(self.indexes) / self.batch_size))  # + 1 to not discard last batch
        return int((len(self.indexes)) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation_hierarchical__(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.indexes_with_user
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation_hierarchical__(self, indexes: list):
        """
        Generates data containing batch_size samples
        Args:
            indexes: ()

        Returns:

        """
        ''  # X : (n_samples, *dim, n_channels)
        user_texts = []

        labels = []
        for user, range_indexes in indexes:
            # PHQ8_Binary
            if 'label' in self.user_level_texts[user]:
                label = self.user_level_texts[user]['label']
            else:
                label = None

            current_batch = np.array([self.user_level_texts[user]['raw'][i] for i in range_indexes])
            user_texts.append(current_batch)
            labels.append(label)

        labels = np.array(labels, dtype=np.float32)
        user_texts = np.array(user_texts, dtype=np.str).reshape(-1, self.max_posts_per_user)

        return user_texts, labels

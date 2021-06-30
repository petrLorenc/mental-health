from utils.logger import logger

from abc import abstractmethod

import pickle
import numpy as np
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.utils import Sequence
from resource_loading import load_NRC, load_vocabulary, load_list_from_file


class AbstractDataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self,
                 user_level_data,
                 subjects_split,
                 set_type,
                 hyperparams_features,
                 batch_size,
                 seq_len,
                 max_posts_per_user=10,
                 shuffle=True,
                 keep_last_batch=True,
                 keep_first_batches=False):
        # Data initialization
        self.set = set_type
        self.data = user_level_data
        self.subjects_split = subjects_split

        # Hyperparameter for training
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_posts_per_user = max_posts_per_user
        self.keep_first_batches = keep_first_batches  # in the rolling window case, whether it will keep
        self.keep_last_batch = keep_last_batch
        self.shuffle = shuffle
        self.padding = "pre"
        self.pad_value = 0

        # Initialization of utils
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.indexes = []
        self.indexes_per_user = {u: [] for u in self.subjects_split[self.set] if u in self.data and len(self.data[u]["texts"]) > 0}
        self.indexes_with_user = []

        self.on_data_loaded()

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = self.indexes_with_user
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def on_data_loaded(self):
        for u in self.indexes_per_user.keys():
            data = self.data[u]
            user_posts = data['texts']

            # Rolling window of datapoints: chunks with overlapping posts
            nr_post_groups = len(user_posts)

            if self.keep_first_batches:
                # Generate datapoints for first posts, before a complete chunk
                for i in range(1, min(self.max_posts_per_user, nr_post_groups - 1)):
                    self.indexes_per_user[u].append(range(0, i))
                    self.indexes_with_user.append((u, range(0, i)))
            elif nr_post_groups < self.max_posts_per_user:
                self.indexes_per_user[u].append(range(0, nr_post_groups))
                self.indexes_with_user.append((u, range(0, nr_post_groups)))

            for i in range(nr_post_groups):
                # Stop at the last complete chunk
                if i + self.max_posts_per_user > len(user_posts):
                    break
                self.indexes_per_user[u].append(range(i, min(i + self.max_posts_per_user, len(user_posts))))
                self.indexes_with_user.append((u, range(i, min(i + self.max_posts_per_user, len(user_posts)))))
        self.on_epoch_end()

    def yield_data_grouped_by_users(self):
        frozen_users = list(self.indexes_per_user.keys())
        for user in frozen_users:
            logger.debug(f"Testing {user} user")
            yield self.get_data_for_specific_user(user)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        features = []

        labels = []
        for user, range_indexes in indexes:
            # PHQ8 binary
            labels.append(self.data[user]['label'] if "label" in self.data[user] else None)
            # Get features
            features.append(self.get_features_for_user_in_data_range(user, range_indexes))

        labels = np.array(labels, dtype=np.float32)
        try:
            features = np.array(features)
        except:
            pass
        # user_texts = np.array(user_texts, dtype=np.str).reshape(-1, self.max_posts_per_user)
        return features, labels


    @abstractmethod
    def get_features_for_user_in_data_range(self, user, data_range):
        pass

    @abstractmethod
    def get_data_for_specific_user(self, user):
        pass

    @abstractmethod
    def __len__(self):
        pass



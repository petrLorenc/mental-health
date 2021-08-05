from utils.logger import logger

from abc import abstractmethod

import numpy as np
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.utils import Sequence


class AbstractDataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self,
                 user_level_data,
                 subjects_split,
                 set_type,
                 batch_size,
                 max_seq_len,
                 chunk_size,
                 data_generator_id,
                 shuffle=False,
                 keep_last_batch=True,
                 keep_first_batches=False):
        # Data initialization
        self.set = set_type
        self.data = user_level_data
        self.subjects_split = subjects_split

        # Hyperparameter for training
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.chunk_size = chunk_size
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
        self.data_generator_id = data_generator_id

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
                for i in range(1, min(self.chunk_size, nr_post_groups - 1)):
                    self.indexes_per_user[u].append(range(0, i))
                    self.indexes_with_user.append((u, range(0, i)))
            elif nr_post_groups < self.chunk_size:
                self.indexes_per_user[u].append(range(0, nr_post_groups))
                self.indexes_with_user.append((u, range(0, nr_post_groups)))

            for i in range(nr_post_groups):
                # Stop at the last complete chunk
                if i + self.chunk_size > nr_post_groups:
                    break
                self.indexes_per_user[u].append(range(i, min(i + self.chunk_size, nr_post_groups)))
                self.indexes_with_user.append((u, range(i, min(i + self.chunk_size, nr_post_groups))))
        self.on_epoch_end()

    def yield_data_grouped_by_users(self):
        frozen_users = list(self.indexes_per_user.keys())
        for user in frozen_users:
            # logger.debug(f"{self.data_generator_id} generator generate data for {user} user")
            yield self.get_data_for_specific_user(user), self.get_label_for_specific_user(user), self.get_text_for_specific_user(user)

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

        labels = np.array( labels, dtype=np.float32)
        try:
            features = np.array(features, dtype=np.float32)
        except:
            try:
                features = np.array(features, dtype=np.str)
            except:
                pass

        # user_texts = np.array(user_texts, dtype=np.str).reshape(-1, self.chunk_size)
        return features, labels

    def get_label_for_specific_user(self, user):
        return self.data[user]["label"]

    def get_text_for_specific_user(self, user):
        return "\n".join(self.data[user]["raw"])

    @abstractmethod
    def get_features_for_user_in_data_range(self, user, data_range):
        pass

    @abstractmethod
    def get_data_for_specific_user(self, user):
        pass

    def __len__(self):
        if self.keep_last_batch:
            return int(np.ceil(len(self.indexes) / self.batch_size))  # + 1 to not discard last batch
        return int((len(self.indexes)) / self.batch_size)



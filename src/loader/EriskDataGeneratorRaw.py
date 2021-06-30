from utils.logger import logger

import numpy as np

from loader.AbstractDataGenerator import AbstractDataGenerator


class EriskDataGeneratorRaw(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, hyperparams_features, batch_size, seq_len,
                 max_posts_per_user=10, shuffle=True, keep_last_batch=True, keep_first_batches=False):

        super().__init__(user_level_data, subjects_split, set_type, hyperparams_features, batch_size, seq_len, max_posts_per_user, shuffle,
                         keep_last_batch, keep_first_batches)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        if self.keep_last_batch:
            return int(np.ceil(len(self.indexes) / self.batch_size))  # + 1 to not discard last batch
        return int((len(self.indexes)) / self.batch_size)

    def get_features_for_user_in_data_range(self, user, data_range):
        current_batch = np.array([self.data[user]['raw'][i] for i in data_range], dtype=str)
        return current_batch

    def get_data_for_specific_user(self, user):
        label = self.data[user]['label']
        user_texts = []
        for indexes in self.indexes_per_user[user]:
            user_texts.append([self.data[user]['raw'][i] for i in indexes])

        return np.array(user_texts, dtype=np.str), label, " ".join([" ".join(x) for x in user_texts])


from utils.logger import logger

import numpy as np

from loader.AbstractDataGenerator import AbstractDataGenerator


class DataGeneratorRaw(AbstractDataGenerator):
    """Generates data for Keras"""

    def get_features_for_user_in_data_range(self, user, data_range):
        current_batch = np.array([self.data[user]['raw'][i] for i in data_range], dtype=str)
        return current_batch

    def get_data_for_specific_user(self, user):
        label = self.data[user]['label']
        user_texts = []
        for indexes in self.indexes_per_user[user]:
            user_texts.append([self.data[user]['raw'][i] for i in indexes])

        return np.array(user_texts, dtype=np.str), label, " ".join([" ".join(x) for x in user_texts])


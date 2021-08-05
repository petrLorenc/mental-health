from utils.logger import logger
import random
import numpy as np

from loader.AbstractDataGenerator import AbstractDataGenerator


class DataGeneratorStr(AbstractDataGenerator):
    """Generates data for Keras"""

    def get_features_for_user_in_data_range(self, user, data_range):
        current_batch = [self.data[user]['raw'][i] for i in data_range]
        if len(current_batch) < self.chunk_size:
            for _ in range(0, self.chunk_size - len(current_batch)):
                current_batch.insert(0, "")
        current_batch = np.array(current_batch, dtype=str)
        return current_batch

    def get_data_for_specific_user(self, user):
        for indexes in self.indexes_per_user[user]:
            raw_texts = [self.data[user]['raw'][i] for i in indexes]

            for _ in range(0, self.chunk_size - len(raw_texts)):
                raw_texts.insert(0, "")

            yield np.array(raw_texts, dtype=np.str).reshape(1, -1)


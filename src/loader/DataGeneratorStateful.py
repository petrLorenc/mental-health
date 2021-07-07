from utils.logger import logger

import numpy as np

from loader.AbstractDataGenerator import AbstractDataGenerator
from tensorflow.keras.preprocessing import sequence


class DataGeneratorStateful(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, batch_size, data_generator_id, vectorizer, shuffle=True):
        self.vectorizer = vectorizer
        super().__init__(user_level_data=user_level_data, subjects_split=subjects_split, set_type=set_type, batch_size=batch_size,
                         seq_len=None, max_posts_per_user=None, data_generator_id=data_generator_id, shuffle=shuffle)

    def on_epoch_end(self):
        self.frozen_users = list(self.indexes_per_user.keys())
        if self.shuffle:
            np.random.shuffle(self.frozen_users)

    # def __len__(self):
    #     return len(self.indexes_per_user)
    #
    # def __getitem__(self, index):
    #     """Generate one batch of data"""
    #     # Generate indexes of the batch
    #     user = self.frozen_users[index]
    #     label = self.data[user]['label']
    #     labels = np.array([label])
    #     features = []
    #
    #     labels = []
    #     for user, range_indexes in indexes:
    #         # PHQ8 binary
    #         labels.append(self.data[user]['label'] if "label" in self.data[user] else None)
    #         # Get features
    #         features.append(self.get_features_for_user_in_data_range(user, range_indexes))
    #
    #     labels = np.array(labels, dtype=np.float32)
    #     try:
    #         features = np.array(features, dtype=np.float32)
    #     except:
    #         pass
    #     # user_texts = np.array(user_texts, dtype=np.str).reshape(-1, self.max_posts_per_user)
    #     return features, labels

    def on_data_loaded(self):
        pass

    def get_data_for_specific_user(self, user):
        label = self.data[user]['label']
        raw_text_array = self.data[user]['raw']

        return np.array(self.vectorizer(raw_text_array).numpy(), dtype=np.float32).reshape((-1, 1, 512)), label, " ".join(raw_text_array)


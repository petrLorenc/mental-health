from utils.logger import logger

import numpy as np

from loader.AbstractDataGenerator import AbstractDataGenerator
from tensorflow.keras.preprocessing import sequence

class DataGeneratorUSE(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, batch_size, seq_len, max_posts_per_user, data_generator_id, vectorizer, shuffle):
        self.vectorizer = vectorizer
        super().__init__(user_level_data=user_level_data, subjects_split=subjects_split, set_type=set_type, batch_size=batch_size,
                         seq_len=seq_len, max_posts_per_user=max_posts_per_user, data_generator_id=data_generator_id, shuffle=shuffle)

    def get_features_for_user_in_data_range(self, user, data_range):
        user_texts = [self.data[user]['raw'][i] for i in data_range]

        for _ in range(0, self.max_posts_per_user - len(user_texts)):
            user_texts.insert(0, "PAD")

        current_batch = self.vectorizer(user_texts).numpy()
        return current_batch

    def get_data_for_specific_user(self, user):
        label = self.data[user]['label']
        user_texts = []
        raw_texts = []
        for indexes in self.indexes_per_user[user]:
            raw_text_array = [self.data[user]['raw'][i] for i in indexes]
            for _ in range(0,  self.max_posts_per_user - len(raw_text_array)):
                raw_text_array.insert(0, "PAD")

            raw_texts.append(" ".join(raw_text_array))
            user_texts.append(self.vectorizer(raw_text_array).numpy())

        # tokens_data_padded = np.array(sequence.pad_sequences(f_tokens, maxlen=self.seq_len,
        #                                                      padding=self.padding,
        #                                                      truncating=self.padding))

        return np.array(user_texts, dtype=np.float32).reshape((-1, self.max_posts_per_user, 512)), label, raw_texts


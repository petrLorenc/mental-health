from utils.logger import logger

import numpy as np

from loader.AbstractDataGenerator import AbstractDataGenerator


class DataGeneratorVectorDistilBERT(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, batch_size, seq_len, max_posts_per_user, data_generator_id, vectorizer, shuffle,
                 embedding_dimension):
        self.vectorizer = vectorizer
        self.embedding_dimension = embedding_dimension
        super().__init__(user_level_data=user_level_data, subjects_split=subjects_split, set_type=set_type, batch_size=batch_size,
                         seq_len=seq_len, max_posts_per_user=max_posts_per_user, data_generator_id=data_generator_id, shuffle=shuffle)

    def get_features_for_user_in_data_range(self, user, data_range):
        user_texts = [self.data[user]['raw'][i] for i in data_range]

        if len(user_texts) == 0:
            return np.zeros(shape=(self.max_posts_per_user, self.embedding_dimension))

        # extract CLS token
        current_batch = np.array([np.squeeze(self.vectorizer(text, pad_to_max_length=True))[0] for text in user_texts])
        # padding with zeros
        current_batch.resize((self.max_posts_per_user, self.embedding_dimension), refcheck=False)

        return current_batch

    def get_data_for_specific_user(self, user):
        for indexes in self.indexes_per_user[user]:
            raw_text_array = [self.data[user]['raw'][i] for i in indexes]

            if len(raw_text_array) == 0:
                yield np.zeros(shape=(1, self.max_posts_per_user, self.embedding_dimension))
            else:
                o = self.vectorizer(raw_text_array).numpy()
                o.resize((1, self.max_posts_per_user, self.embedding_dimension), refcheck=False)
                yield np.array(o, dtype=np.float32)

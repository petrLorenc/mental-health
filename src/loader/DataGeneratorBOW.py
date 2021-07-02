from utils.logger import logger

from sklearn.feature_extraction.text import CountVectorizer
from loader.AbstractDataGenerator import AbstractDataGenerator
from resource_loading import load_list_from_file


class DataGeneratorBow(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, hyperparams_features, batch_size, vectorizer=None, data_generator_id=""):

        if vectorizer is None:
            vectorizer_vocabulary = load_list_from_file(hyperparams_features["vocabulary_path"])
            self.vectorizer = CountVectorizer(vocabulary=vectorizer_vocabulary)
        else:
            self.vectorizer = vectorizer

        super().__init__(user_level_data=user_level_data, subjects_split=subjects_split, set_type=set_type, batch_size=batch_size,
                         seq_len=None, max_posts_per_user=None, shuffle=False,
                         keep_last_batch=True, keep_first_batches=True, data_generator_id=data_generator_id)

    def on_data_loaded(self):
        for u in self.subjects_split[self.set]:
            if u in self.data:
                self.indexes_with_user.append((u, None))

        self.on_epoch_end()

    def get_features_for_user_in_data_range(self, user, data_range):
        # data_range not used for BoW
        user_texts = " ".join([x for x in self.data[user]["raw"]])

        return self.vectorizer.transform([user_texts]).toarray()

    def get_data_for_specific_user(self, user):
        label = self.data[user]['label']
        user_texts = " ".join([x for x in self.data[user]["raw"]])

        return self.vectorizer.transform([user_texts]).toarray(), label, user_texts

    def __len__(self):
        return len(self.indexes_per_user) - 1

    def get_input_dimension(self):
        if self.vectorizer is not None:
            return len(self.vectorizer.get_feature_names())
        else:
            raise RuntimeError("Vectorizer is not set yet")
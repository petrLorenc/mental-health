import json
import string
import random
from nltk.tokenize import RegexpTokenizer
from loader.DataGenerator import DataGenerator


class DAICDataGenerator(DataGenerator):
    def __init__(self, test_data_object, nickname=None, **kwargs):
        self.data = {}
        self.subjects_split = {'test': []}
        self.tokenizer = RegexpTokenizer(r'\w+')
        super().__init__(self.data, self.subjects_split, set_type='test', **kwargs)
        if test_data_object is not None:
            self.prepare_data(test_data_object, nickname=nickname)

    def load_daic_data(self, test_data_object, nickname=None):
        subjects_split = {'test': []}
        user_level_texts = {}

        session = test_data_object
        if nickname is not None:
            nickname = str(nickname)
        else:
            nickname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
        for datapoint in session["transcripts"]:
            words = []
            raw_text = ""
            if datapoint["speaker"] == "Participant":
                if "value" in datapoint:
                    tokenized_text = self.tokenizer.tokenize(datapoint["value"])
                    words.extend(tokenized_text)
                    raw_text += datapoint["value"]

                if nickname not in user_level_texts.keys():
                    user_level_texts[nickname] = {}
                    subjects_split['test'].append(nickname)
                    user_level_texts[nickname]['texts'] = []
                    user_level_texts[nickname]['raw'] = []
                    user_level_texts[nickname]["label"] = []

                user_level_texts[nickname]['texts'].append(words)
                user_level_texts[nickname]['raw'].append(raw_text)
                user_level_texts[nickname]["label"].append(int(session["label"]["PHQ8_Binary"]))


        return user_level_texts, subjects_split

    def prepare_data(self, test_data_object, nickname=None):
        user_level_texts, subjects_split = self.load_daic_data(test_data_object, nickname=nickname)
        for u in user_level_texts:
            if u not in self.data:
                self.data[u] = {k: [] for k in user_level_texts[u].keys()}
            for k in user_level_texts[u].keys():
                self.data[u][k].extend(user_level_texts[u][k])
        self.subjects_split['test'].extend(subjects_split['test'])
        self.subjects_split['test'] = list(set(self.subjects_split['test']))
        self._post_indexes_per_user()
        self.on_epoch_end()
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Reset generated labels
        if index == 0:
            self.generated_labels = []
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find users
        user_indexes = [t[0] for t in indexes]
        users = set([self.subjects_split[self.set][i] for i in user_indexes
                     if self.subjects_split[self.set][
                         i] in self.data.keys()])  # TODO: maybe needs a warning that user is missing
        post_indexes_per_user = {u: [] for u in users}
        # Sample post ids
        for u, post_indexes in indexes:
            user = self.subjects_split[self.set][u]
            # Note: was bug here - changed it into a list
            post_indexes_per_user[user].append(post_indexes)

        X, s, y = self.__data_generation_hierarchical__(users, post_indexes_per_user)
        if self.return_subjects:
            return X, s, y
        else:
            return X, y
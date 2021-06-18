import json
from nltk.tokenize import RegexpTokenizer
from src.loader.DataGenerator import DataGenerator
import logging
from src.loader.data_loading import load_erisk_server_data


class DAICDataGenerator(DataGenerator):
    def __init__(self, test_data_object, idx=0, **kwargs):
        self.data = {}
        self.subjects_split = {'test': []}
        self.tokenizer = RegexpTokenizer(r'\w+')
        if 'logger' in kwargs:
            self.logger = kwargs['logger']
        else:
            self.logger = None
        super().__init__(self.data, self.subjects_split, set_type='test', logger=self.logger, **kwargs)
        self.idx = idx
        self.prepare_data(test_data_object)

    def load_daic_data(self, test_data_object):
        subjects_split = {'test': []}
        user_level_texts = {}

        session = test_data_object
        nickame = str(self.idx)
        for datapoint in session["transcripts"]:
            words = []
            raw_text = ""
            if datapoint["speaker"] == "Participant":
                if "value" in datapoint:
                    tokenized_text = self.tokenizer.tokenize(datapoint["value"])
                    words.extend(tokenized_text)
                    raw_text += datapoint["value"]

                if nickame not in user_level_texts.keys():
                    user_level_texts[nickame] = {}
                    user_level_texts[nickame]['texts'] = [words]
                    user_level_texts[nickame]['raw'] = [raw_text]
                    subjects_split['test'].append(nickame)
                else:
                    user_level_texts[nickame]['texts'].append(words)
                    user_level_texts[nickame]['raw'].append(raw_text)

        return user_level_texts, subjects_split

    def prepare_data(self, test_data_object):
        user_level_texts, subjects_split = self.load_daic_data(test_data_object)
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
        if len(self.data) == 0:
            if self.logger:
                self.logger.error("Cannot generate with zero data.\n")
            return
        if len(self.data) <  self.posts_per_group:
            if self.logger:
                self.logger.warning("Number of input datapoints (%d) lower than minimum number of posts per chunk (%d).\n" % (len(self.data), self.posts_per_group))
        return super().__getitem__(index)
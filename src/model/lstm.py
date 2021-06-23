import json
import numpy as np

import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Input
from tensorflow.keras.utils import to_categorical


module_url = "../resources/embedding/use-4"
model = hub.load(module_url)

# class DummyModel:
#     def __call__(self, *args, **kwargs):
#         return np.random.random(3)
# model = DummyModel()


class LSTM_Network:
    def __init__(self):
        self.dim = 128
        self.embeddings_size = 512

        self.model = self.get_model()

    def get_model(self):
        model = Sequential()
        # model.add(TimeDistributed(hub.KerasLayer(module_url, dtype=tf.string, trainable=True, name="use", output_shape=(None, 512)), input_shape=(None, None, 1)))
        model.add(LSTM(self.dim, return_sequences=False, input_shape=(None, self.embeddings_size)))
        model.add(Dense(2, activation='sigmoid'))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def train_model(self, train_json):
        X = []
        y = []
        for item in train_json:
            sentences = []
            for transcript in item["transcripts"]:
                sentences.append(model([transcript["value"]]).numpy())

            for i in range(1, len(sentences)):
                temp_X = []
                for ii in range(0, i):
                    temp_X.append(sentences[ii])
                if len(temp_X) > 0:
                    X.append(temp_X)
                    y.append([0, 1] if item["label"]["PHQ8_Binary"] == '0' else [1, 0])

        X = np.asarray(X)
        y = np.asarray(y)
        print(X.shape)
        print(y.shape)

        def train_generator():
            for _x, _y in zip(X, y):
                # sequence_length = np.random.randint(10, 100)
                # x_train = np.random.random((1000, sequence_length, 5))
                # # y_train will depend on past 5 timesteps of x
                # y_train = x_train[:, :, 0]
                # for i in range(1, 5):
                #     y_train[:, i:] += x_train[:, :-i, i]
                # y_train = to_categorical(y_train > 2.5)
                yield np.asarray(_x).reshape((1, -1, self.embeddings_size)), np.asarray(_y).reshape((1, 2))

        self.model.fit_generator(train_generator(), steps_per_epoch=len(X), epochs=3, verbose=1)
        # self.model.fit(X, y, batch_size=1, epochs=10, shuffle=False)

    def predict(self, inp):
        return self.model.predict(inp)

    def test(self, test_json):
        X = []
        y = []
        for item in test_json:
            sentences = []
            for transcript in item["transcripts"]:
                sentences.append(model([transcript["value"]]).numpy())
            X.append(sentences)
            y.append(1 if item["label"]["PHQ_Binary"] == '0' else 0)

        X = np.asarray(X)
        y = np.asarray(y)

        acc = []
        for _x, _y in zip(X, y):
            if len(_x) > 0:
                pred_y = self.model.predict(np.asarray(_x).reshape((1, -1, self.embeddings_size)))[0]
                pred_cls = np.argmax(pred_y)
                acc.append(int(_y == pred_cls))

        print(f"Accuracy: {sum(acc) / len(acc)}")


if __name__ == '__main__':
    with open("../data/daic-woz/train_data.json", "r") as f:
        train_json = json.load(f)
    lstm = LSTM_Network()
    lstm.train_model(train_json)
    print(lstm.predict(np.random.random((1, 30, 3))))

    with open("../data/daic-woz/test_data.json", "r") as f:
        test_json = json.load(f)
    lstm.test(test_json)

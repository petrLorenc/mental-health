from collections.abc import MutableMapping


class DefaultHyperparameters(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = {
            "positive_class_weight": 2,
            "chunk_size": 15,
            "batch_size": 8,
            "epochs": 50,
            "max_seq_len": 50,

            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 2,
            "scheduled_reduce_lr_freq": 2,
            "scheduled_reduce_lr_factor": 0.5,
            "early_stopping_patience": 10,
            "learning_rate": 0.001,
            "threshold": 0.5,

            "optimizer": "adam",
        }
        self.store.update(**kwargs)
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return f"{type(self).__name__}({self.store})"


class DefaultHyperparametersSequence(DefaultHyperparameters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.store.update({
            "dropout": 0.1,
            "l2_dense": 0.00011,
            "l2_embeddings": 1e-07,
            "norm_momentum": 0.1,
            "ignore_layer": [],
            "lstm_units_user": 100,
            "lstm_units": 64,
            "decay": 0.001,
            "lr": 5e-05,
            "padding": "pre"
        })
        self.update(dict(*args, **kwargs))  # use the free update to set keys


hyperparams = DefaultHyperparameters()

if __name__ == '__main__':
    default_hyperparameters = DefaultHyperparameters({"xx": 33})
    print(default_hyperparameters)
    print(default_hyperparameters["xx"])

from utils.logger import logger

from tensorflow.keras import callbacks
import tensorflow.keras.backend as K


class WeightsHistory(callbacks.Callback):
    def __init__(self, experiment, logs={}):
        super(WeightsHistory, self).__init__()
        self.experiment = experiment

    def on_train_begin(self, logs={}):
        self.log_weights(0)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
            self.log_weights(epoch)

    def log_weights(self, step):
        for layer in self.model.layers:
            try:
                self.experiment.log_histogram_3d(layer.get_weights()[0], name=layer.name + "_weight", step=step)
            except Exception as e:
                # logger.debug("Logging weights error: " + layer.name + "; " + str(e) + "\n")
                pass


class LRHistory(callbacks.Callback):
    def __init__(self, experiment, logs={}):
        super(LRHistory, self).__init__()
        self.experiment = experiment

    def on_epoch_begin(self, epoch, logs={}):
        self.log_lr()

    def log_lr(self):
        lr = K.eval(self.model.optimizer.lr)
        logger.debug("Learning rate is %f...\n" % lr)
        self.experiment.log_parameter('lr', lr)


# todo
class FreezeLayer(callbacks.Callback):
    def __init__(self, experiment, logs={}, patience=5, layer={'user_encoder': 'embeddings_layer'}, verbose=1, set_to=False):
        super(FreezeLayer, self).__init__()
        self.freeze_epoch = patience
        self.freeze_layer = layer
        self.verbose = verbose
        self.set_to = set_to
        self.experiment = experiment

    def on_epoch_begin(self, epoch, logs={}):
        if type(self.freeze_layer) == dict:
            submodel = self.model.get_layer(list(self.freeze_layer.keys())[0])
        else:
            submodel = self.model
        logger.debug("Trainable embeddings", submodel.get_layer(self.freeze_layer).trainable)
        if epoch == self.freeze_epoch:
            try:
                layer = submodel.get_layer(self.freeze_layer)
                old_value = layer.trainable
                layer.trainable = self.set_to
                # TODO: does this reset the optimizer? should I also compile the top-level model?
                self.model.compile(hyperparams['optimizer'], binary_crossentropy_custom,
                                   metrics=[metrics_class.f1_m, metrics_class.precision_m, metrics_class.recall_m])
                if self.verbose:
                    logger.debug("Setting %s layer from %s to trainable=%s...\n" % (layer.name, old_value,
                                                                                     submodel.get_layer(
                                                                                         self.freeze_layer).trainable))
            except Exception as e:
                # layer probably does not exist
                pass

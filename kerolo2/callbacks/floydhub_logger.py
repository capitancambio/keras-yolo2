from keras.callbacks import Callback


class MAPCallback(Callback):

    def __init__(self, train_generator, validation_generator, yolo):
        super(MAPCallback, self).__init__()
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.yolo = yolo

    def on_epoch_end(self, epoch, logs=None):
        average_precisions = self.yolo.evaluate(self.train_generator)
        for label, average_precision in average_precisions.items():
            label = self.yolo.labels[label]
            logs[label] = average_precision
        mAP = sum(average_precisions.values()) / len(average_precisions)
        logs['mAP'] = mAP

        average_precisions = self.yolo.evaluate(self.validation_generator)
        for label, average_precision in average_precisions.items():
            label = self.yolo.labels[label]
            logs["val_"+label] = average_precision
        mAP = sum(average_precisions.values()) / len(average_precisions)
        logs['val_mAP'] = mAP
        return logs


class FloydhubLoggerCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        for metric, value in logs.items():
            print('{{"metric": "{}", "value": {}}}'.format(metric, value))

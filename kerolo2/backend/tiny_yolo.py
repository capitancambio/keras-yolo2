from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from .backend import BaseFeatureExtractor

import logging

logger = logging.getLogger(__name__)


class TinyYoloFeature(BaseFeatureExtractor):
    """Full yolo feature implementation """

    def _build_model(self):
        logger.info("Building tiny tolo")
        input_image = Input(shape=(self.input_size, self.input_size, 3))

        # Layer 1
        x = Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                   name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0, 4):
            x = Conv2D(32*(2**i), (3, 3), strides=(1, 1), padding='same',
                       name='conv_' + str(i+2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                   name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

        # Layer 7 - 8
        for i in range(0, 2):
            x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same',
                       name='conv_' + str(i+7), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+7))(x)
            x = LeakyReLU(alpha=0.1)(x)

        model = Model(input_image, x)
        model.load_weights(self.weight_file)
        return model

    def normalize(self, image):
        return image / 255.

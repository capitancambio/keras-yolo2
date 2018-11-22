from keras.models import Model
from keras.layers import Input
from keras.applications.mobilenet import MobileNet

from .backend import BaseFeatureExtractor
import logging

logger = logging.getLogger(__name__)


class MobileNetFeature(BaseFeatureExtractor):

    """Movile net model"""

    def _build_model(self):
        logger.info("Building mobilenet...")
        print("building mobilenet")
        input_image = Input(shape=(self.input_size, self.input_size, 3))

        mobilenet = MobileNet(input_shape=(224, 224, 3), include_top=False)

        mobilenet.load_weights(self.weight_file)

        x = mobilenet(input_image)

        return Model(input_image, x)

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image

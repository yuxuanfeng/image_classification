from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np


class DeepFeature:
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)
        self.model.summary()

    def get_feature(self, image_file):
        img = image.load_img(image_file, target_size=(224,224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        img_feature = self.model.predict(img_data)
        return img_feature


<<<<<<< HEAD
from time import time
import os
import cv2
from PIL import Image
=======
import cv2
import time
import os
from PIL import Image
from typing import Tuple
>>>>>>> adf14ee3f7f847b72cf784572a61bcfcd7fe8fc8
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T
from database.read_database import ReadImages


<<<<<<< HEAD

=======
>>>>>>> adf14ee3f7f847b72cf784572a61bcfcd7fe8fc8
class AutoLabellingObjectDetect:
    def __init__(self):
        self.data = ReadImages()

    def main(self):
<<<<<<< HEAD
        images, names = self.data.read_images('database/untagged_images')
        print(len(images), len(names))

        # main process

=======
        images, names = self.data.read_images('database/images_with_labels')
        print(len(images), len(names))

>>>>>>> adf14ee3f7f847b72cf784572a61bcfcd7fe8fc8

auto_labeling = AutoLabellingObjectDetect()
auto_labeling.main()
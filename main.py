import cv2
import time
import os
from PIL import Image
from typing import Tuple
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T
from database.read_database import ReadImages


class AutoLabellingObjectDetect:
    def __init__(self):
        self.data = ReadImages()

    def main(self):
        images, names = self.data.read_images('database/images_with_labels')
        print(len(images), len(names))


auto_labeling = AutoLabellingObjectDetect()
auto_labeling.main()
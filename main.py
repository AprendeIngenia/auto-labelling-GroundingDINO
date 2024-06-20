<<<<<<< HEAD
<<<<<<< HEAD
import time
from database.read_database import ReadImages
=======
from database import ReadImages
>>>>>>> a5ca029 (initial commit)
=======
from time import time
import os
import cv2
from PIL import Image
from database.read_database import ReadImages
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T

>>>>>>> 66ff857 (add requirements)


class AutoLabellingObjectDetect:
    def __init__(self):
        self.data = ReadImages()

    def main(self):
        images, names = self.data.read_images('database/untagged_images')
        print(len(images), len(names))

        # main process


auto_labeling = AutoLabellingObjectDetect()
auto_labeling.main()
import time
from database.read_database import ReadImages


class AutoLabellingObjectDetect:
    def __init__(self):
        self.data = ReadImages()

    def main(self):
        images, names = self.data.read_images('database/images_with_labels')
        print(len(images), len(names))


auto_labeling = AutoLabellingObjectDetect()
auto_labeling.main()

# hola 
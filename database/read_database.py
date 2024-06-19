import os
import cv2


class ReadImages:
    def __init__(self):
        self.images: list = []
        self.names: list = []

    def read_images(self, database_path: str):
        data = os.listdir(database_path)

        for lis in data:
            imgdb = cv2.imread(f'{database_path}/{lis}')
            self.images.append(imgdb)
            self.names.append(os.path.splitext(lis)[0])

        return self.images, self.names
from database import ReadImages


class AutoLabellingObjectDetect:
    def __init__(self):
        self.data = ReadImages()

    def main(self):
        images, names = self.data.read_images('database/images_with_labels')
        print(len(images), len(names))

        # main process


auto_labeling = AutoLabellingObjectDetect()
auto_labeling.main()
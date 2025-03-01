from time import time
import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Any
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T
from database.read_database import ReadImages


class AutoLabellingObjectDetect:
    def __init__(self):
        self.data = ReadImages()

        # variables
        self.cont: int = 0
        self.num_images: int = 0
        self.class_id: int = 2

        self.box_threshold: float = 0.38
        self.text_threshold: float = 0.25

        self.out_path: str = 'database/tagged_images/'
        self.prompt: str = 'face'
        self.home: str = os.getcwd()

        self.save: bool = True
        self.draw: bool = False

        self.images: list = []
        self.names: list = []
        self.bbox_info: list = []

    def save_data(self, image_copy: np.ndarray, list_info: list):
        timeNow = time()
        timeNow = str(timeNow)
        timeNow = timeNow.split('.')
        timeNow = timeNow[0] + timeNow[1]
        cv2.imwrite(f"{self.out_path}/{timeNow}.jpg", image_copy)
        for info in list_info:
            f = open(f"{self.out_path}/{timeNow}.txt", 'a')
            f.write(info)
            f.close()

    def config_grounding_model(self) -> Any:
        config_path = os.path.join(self.home, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        check_point_path = 'GroundingDINO/weights/groundingdino_swint_ogc.pth'
        model = load_model(config_path, check_point_path, device="cuda")
        return model

    def main(self):
        self.images, self.names = self.data.read_images('database/untagged_images')
        self.num_images = len(self.images)
        grounding_model = self.config_grounding_model()

        while self.cont < self.num_images:
            self.bbox_info = []
            print('------------------------------------')
            print(f'name_image: {self.names[self.cont]}')

            process_image = self.images[self.cont]
            copy_image = process_image.copy()
            draw_image = process_image.copy()

            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )

            img_source = Image.fromarray(process_image).convert("RGB")
            img_transform, _ = transform(img_source, None)

            boxes, logits, phrases = predict(
                model=grounding_model,
                image=img_transform,
                caption=self.prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device="cuda"
            )

            if len(boxes) != 0:
                h, w, _ = process_image.shape
                for i, box in enumerate(boxes):
                    xc, yc, an, al = box[0], box[1], box[2], box[3]

                    xc = max(0, min(xc, 1))
                    yc = max(0, min(yc, 1))
                    an = max(0, min(an, 1))
                    al = max(0, min(al, 1))

                    self.bbox_info.append(f"{self.class_id} {xc} {yc} {an} {al}\n")
                    x1, y1, x2, y2 = int(xc * w), int(yc * h), int(an * w), int(al * h)
                    print(f"boxes: {boxes}\nxc: {x1} yc:{y1} w:{x2} h:{y2}")

                if self.save:
                    self.save_data(copy_image, self.bbox_info)

                if self.draw:
                    annotated_img = annotate(image_source=draw_image, boxes=boxes, logits=logits, phrases=phrases)
                    out_frame = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    cv2.imshow('Grounding DINO detect', out_frame)
                    cv2.waitKey(0)

            self.cont += 1


auto_labeling = AutoLabellingObjectDetect()
auto_labeling.main()

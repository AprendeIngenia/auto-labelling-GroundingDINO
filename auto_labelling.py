import os
import sys
import cv2
import torch
import logging
import numpy as np
from time import time
from PIL import Image
from typing import List, Any
from database.read_database import ReadImages

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

try:
    from GroundingDINO.groundingdino.util.inference import load_model, predict, annotate
    import GroundingDINO.groundingdino.datasets.transforms as T
except ImportError as e:
    print("❌ Error de importación: Asegúrate de estar en la carpeta raíz donde clonaste GroundingDINO.")
    raise e

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class AutoLabellingObjectDetect:
    def __init__(self, target_class_id: int):
        self.data = ReadImages()

        # ontology
        self.ontology = {
            0: {"prompt": "red apple fruit", "name": "apple"},
            1: {"prompt": "orange carrot vegetable", "name": "zanahoria"},
            2: {"prompt": "sunglasses eyewear", "name": "gafas"},
            3: {"prompt": "baseball cap hat", "name": "gorras"},
            4: {"prompt": "toilet paper roll", "name": "papel_higienico"},
            5: {"prompt": "toothbrush", "name": "cepillo"},
            6: {"prompt": "soda can", "name": "soda"},
            7: {"prompt": "cereal box carton", "name": "cereal"}
        }

        if target_class_id not in self.ontology:
            raise ValueError(f"ID {target_class_id} no existe en la ontología.")

        self.current_class_id = target_class_id
        self.prompt = self.ontology[target_class_id]["prompt"]
        self.class_name = self.ontology[target_class_id]["name"]

        # model config
        self.box_threshold = 0.45
        self.text_threshold = 0.25
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # paths
        self.home = os.getcwd()
        self.input_dir = 'database/untagged_images'
        self.output_dir = 'database/tagged_images'
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        self._ensure_dirs()
        self.model = self._load_dino_model()

        logger.info(f"🤖 Iniciando Auto-Labeler para: {self.class_name} (ID: {self.current_class_id})")
        logger.info(f"📝 Prompt utilizado: '{self.prompt}'")

        self.cont: int = 0
        self.num_images: int = 0
        self.class_id: int = 2

        self.save: bool = True
        self.draw: bool = False

        self.images: list = []
        self.names: list = []
        self.bbox_info: list = []

    def _ensure_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_dino_model(self):
        config_path = os.path.join(self.home, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        check_point_path = os.path.join(self.home, 'GroundingDINO/weights/groundingdino_swint_ogc.pth')

        if not os.path.exists(check_point_path):
            raise FileNotFoundError(f"No encuentro los pesos en: {check_point_path}")

        return load_model(config_path, check_point_path, device=self.device)

    def get_image_files(self) -> List[str]:
        files = []
        for f in os.listdir(self.input_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in self.valid_extensions:
                files.append(os.path.join(self.input_dir, f))
        return sorted(files)

    def normalize_box(self, box):
        return [max(0.0, min(x, 1.0)) for x in box]

    def save_yolo_label(self, filename: str, boxes: torch.Tensor):
        base_name = os.path.splitext(os.path.basename(filename))[0]
        txt_path = os.path.join(self.output_dir, base_name + ".txt")
        img_output_path = os.path.join(self.output_dir,
                                       os.path.basename(filename))

        with open(txt_path, 'w') as f:
            for box in boxes:
                xc, yc, w, h = self.normalize_box(box.tolist())
                f.write(f"{self.current_class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        return img_output_path

    def process(self, visualize: bool = True):
        image_files = self.get_image_files()
        total = len(image_files)

        if total == 0:
            logger.warning(f"⚠️ No se encontraron imágenes en {self.input_dir}")
            return

        logger.info(f"📂 Procesando {total} imágenes...")

        for idx, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)

            try:
                # 1. load images
                image_cv = cv2.imread(img_path)
                if image_cv is None:
                    logger.error(f"❌ Error leyendo imagen (corrupta o path invalido): {filename}")
                    continue

                image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

                transform = T.Compose([
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                image_tensor, _ = transform(image_pil, None)

                # 2. inference
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image_tensor,
                    caption=self.prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device
                )

                # 3. save images
                if boxes.shape[0] == 0:
                    logger.warning(f"⚠️ {idx + 1}/{total} - Sin detecciones en: {filename}")
                    continue

                # save YOLO format
                self.save_yolo_label(img_path, boxes)
                cv2.imwrite(os.path.join(self.output_dir, filename), image_cv)

                logger.info(f"✅ {idx + 1}/{total} - Guardado: {filename} ({len(boxes)} objetos)")

                # 4. visualize
                if visualize:
                    h_img, w_img, _ = image_cv.shape
                    debug_img = image_cv.copy()

                    for box in boxes:
                        xc, yc, w, h = box.tolist()
                        x1 = int((xc - w / 2) * w_img)
                        y1 = int((yc - h / 2) * h_img)
                        x2 = int((xc + w / 2) * w_img)
                        y2 = int((yc + h / 2) * h_img)

                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(debug_img, f"{self.class_name}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.imshow('Supervision (Presiona Q para salir, espacio para siguiente)', debug_img)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        print("🛑 Ejecución detenida por el usuario.")
                        break

            except Exception as e:
                logger.error(f"❌ Excepción crítica procesando {filename}: {str(e)}")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 0:apple, 1:zanahoria, 2:gafas, 3:gorras, 4:papel, 5:cepillo, 6:soda, 7:cereal
    CLASS_TO_PROCESS = 7
    app = AutoLabellingObjectDetect(target_class_id=CLASS_TO_PROCESS)
    app.process(visualize=False)


from pathlib import Path
import cv2
import sys

TESTS_DIR = Path(__file__).resolve().parent
BASE_DIR = TESTS_DIR.parent
sys.path.append(str(BASE_DIR))

from geometry import FacialGeometryPipeline

IMAGE_PATH = TESTS_DIR / "face.jpg" 
img_original = cv2.imread(str(IMAGE_PATH))

if img_original is None:
    raise FileNotFoundError(
        f"Файл не найден по пути: {IMAGE_PATH}\n"
        f"Убедитесь, что 'face.jpg' лежит в папке: {IMAGE_PATH.parent}"
    )

measurer = FacialGeometryPipeline()
res_original = measurer.process_image(img_original)

img_resized = cv2.resize(img_original, (0, 0), fx=0.5, fy=0.5)
res_resized = measurer.process_image(img_resized)

print("Original Image Ratios:", res_original["ratios"])
print("Resized Image Ratios: ", res_resized["ratios"])
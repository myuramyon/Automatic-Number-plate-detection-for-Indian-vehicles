import cv2
from paddleocr import PaddleOCR
import numpy as np

# Path to the image
img_path = 'Search_Image/sample.jpeg'

# Load image
image = cv2.imread(img_path)
if image is None:
    print(f"Failed to load image: {img_path}")
    exit()

# Run PaddleOCR
ocr = PaddleOCR(use_textline_orientation=True, lang='en')
result = ocr.predict(img_path)[0]

# Draw all detected boxes and print recognized texts
rec_texts = result['rec_texts']
rec_scores = result['rec_scores']
rec_polys = result['rec_polys']

image_with_boxes = image.copy()
for poly, text, score in zip(rec_polys, rec_texts, rec_scores):
    pts = np.array(poly).astype(int)
    cv2.polylines(image_with_boxes, [pts], isClosed=True, color=(0,255,0), thickness=2)
    print(f"Detected: '{text}' (score: {score:.2f})")

output_path = 'Search_Image/sample_boxed.jpg'
cv2.imwrite(output_path, image_with_boxes)
print(f"Boxed image saved to {output_path}")

# Find the box with the highest score (likely the number plate)
if rec_scores:
    best_idx = int(np.argmax(rec_scores))
    best_poly = rec_polys[best_idx]
    x_min = int(min([point[0] for point in best_poly]))
    x_max = int(max([point[0] for point in best_poly]))
    y_min = int(min([point[1] for point in best_poly]))
    y_max = int(max([point[1] for point in best_poly]))
    cropped = image[y_min:y_max, x_min:x_max]
    cropped_path = 'Search_Image/sample_cropped.jpg'
    cv2.imwrite(cropped_path, cropped)
    print(f"Cropped image saved to {cropped_path}")
    # OCR on cropped image
    result_cropped = ocr.predict(cropped_path)[0]
    print("OCR result on cropped image:")
    for text, score in zip(result_cropped['rec_texts'], result_cropped['rec_scores']):
        print(f"'{text}' (score: {score:.2f})")
else:
    print("No text boxes detected.")

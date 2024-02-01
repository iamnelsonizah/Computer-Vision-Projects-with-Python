import cv2
import easyocr
import matplotlib.pyplot as plt

def draw_bounding_boxes(image, detections, threshold=0.25):
    for bbox, text, score in detections:
        if score > threshold:
            cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
            cv2.putText(image, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (255, 0, 0), 2)

# read image
image_path = "image/preview.jpg"

img = cv2.imread(image_path)

if img is None:
        raise ValueError("Error loading the image. Please check the file path.")
    
try:
    # instance text detector
    reader = easyocr.Reader(['en'], gpu=False)

    # detect text on image
    text_detections = reader.readtext(img)
    threshold = 0.25

    # # draw bbox and text
    draw_bounding_boxes(img, text_detections, threshold)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
    plt.show()

except Exception as e:
    print(f"Error: {e}")
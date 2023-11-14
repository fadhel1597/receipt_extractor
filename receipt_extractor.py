from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import cv2

model = YOLO("weights/best.pt")

TRprocessor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
TRmodel = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

def detect_objects(image):
    """
    Detect objects in the image using the YOLO model.

    Parameters:
    image (np.array): The loaded image
    """
    # model = YOLO("logs-2/receipt_extractor2/weights/best.pt")
    results = model.predict(image, imgsz=640, conf=0.5, iou=0.5)[0]

    bbox_list = []

    for result in results:
        bbox = result.boxes.xyxy[0].tolist()
        bbox_list.append(bbox)


    return bbox_list

def read_ocr(image):
    """
    Read text from image using the TrOCR model.

    Parameters:
    image (np.array): The loaded image
    """
    pixel_values = TRprocessor(image, return_tensors="pt").pixel_values
    generated_ids = TRmodel.generate(pixel_values)
    generated_text = TRprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def read_image(bboxes, img, tolerance=20):
    """
    Read images of each horizontal line of bounding boxes separately.

    Parameters:
    bboxes (list): List of bounding boxes [(x1, y1, x2, y2), ...]
    img (np.array): The loaded image
    tolerance (int): The tolerance for considering boxes to be on the same line
    """
    # Sort by y-coordinate (horizontal position)
    bboxes.sort(key=lambda bbox: bbox[1])
    
    # Then sort by x-coordinate (vertical position) within each horizontal line
    i = 0
    line_num = 0
    ocr_results = {}
    while i < len(bboxes):
        j = i
        while j < len(bboxes) and abs(bboxes[j][1] - bboxes[i][1]) < tolerance:  # Use the tolerance parameter here
            j += 1
        line_bboxes = sorted(bboxes[i:j])  # Sort boxes on the same line by x-coordinate

        line_texts = []
        for k, (x1, y1, x2, y2) in enumerate(line_bboxes):

            generated_text = read_ocr(img[int(y1):int(y2), int(x1):int(x2)])
            line_texts.append(generated_text)
        
        ocr_results[f'line_num_{line_num}'] = line_texts

        line_num += 1
        i = j
    
    return ocr_results


if __name__ == "__main__":

    image = cv2.imread('data_test/test_images_1.jpg')
    bbox_list = detect_objects(image)
    print(read_image(bbox_list, image))
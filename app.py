from flask import Flask, jsonify, request
from receipt_extractor import detect_objects, read_image
import numpy as np
import base64
import cv2
import logging

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)


@app.route('/extract_receipt', methods=['POST'])
def extract_receipt():
    data = request.get_json()
    encoded_image = data['image']
    decoded_image = base64.b64decode(encoded_image)
    nparr = np.frombuffer(decoded_image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    try:
        bbox_list = detect_objects(image)
        result = read_image(bbox_list, image)
    except Exception as e:
        logging.error(e)
        result = {'error': 'Something went wrong!'}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

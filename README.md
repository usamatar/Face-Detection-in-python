# Face Detection System using OpenCV

## Overview
This is a simple face detection system that uses OpenCV's pre-trained Haar Cascade model to detect faces in an image. The detected faces are highlighted with rectangles and the processed image is displayed and saved.

## Requirements
### Install OpenCV
Before running the script, ensure you have OpenCV installed. You can install it using the following command:
```sh
pip install opencv-python
```

## How It Works
1. Loads a pre-trained face detection model (`haarcascade_frontalface_alt2.xml`).
2. Reads an input image (`test.jpg`).
3. Converts the image to grayscale for better accuracy.
4. Detects faces in the image using OpenCV's `detectMultiScale()` function.
5. Draws rectangles around detected faces.
6. Displays the processed image.
7. Saves the final image as `detected_faces.jpg`.

## Usage
1. Place the `haarcascade_frontalface_alt2.xml` file in the same directory as your script.
2. Ensure that `test.jpg` exists in the same directory.
3. Run the script:
   ```sh
   python face-detection.py
   ```
4. The detected faces will be displayed and saved as `detected_faces.jpg`.

## Code Explanation
```python
import cv2  # Import OpenCV

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Read the image
img = cv2.imread('test.jpg')

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the processed image
cv2.imwrite('detected_faces.jpg', img)
```

## Troubleshooting
- **ModuleNotFoundError: No module named 'cv2'**
  - Install OpenCV using: `pip install opencv-python`
- **No image displayed or saved**
  - Ensure `test.jpg` is in the same directory.
- **No faces detected**
  - Try adjusting `scaleFactor` and `minNeighbors` values in `detectMultiScale()`.
- **Cascade file not found**
  - Download `haarcascade_frontalface_alt2.xml` from [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades).

## License
This project is open-source and free to use under the MIT License.

## Author
Developed by [Your Name].


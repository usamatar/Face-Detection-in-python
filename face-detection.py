import cv2

# Load the face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Read the image
img = cv2.imread('test.jpg')

# Convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4)

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display image with detected faces
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the processed image
cv2.imwrite('detected_faces.jpg', img)

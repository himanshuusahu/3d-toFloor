import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use Hough Transform to detect lines (potential walls)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title('Edges and Detected Lines')
    plt.imshow(edges, cmap='gray')
    plt.show()
    
    return edges, lines

def create_floor_plan(edges, lines):
    # Create a blank image for the floor plan
    floor_plan = np.ones_like(edges) * 255
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(floor_plan, (x1, y1), (x2, y2), 0, 2)
    
    # Display the floor plan
    plt.figure(figsize=(5, 5))
    plt.title('Floor Plan')
    plt.imshow(floor_plan, cmap='gray')
    plt.show()
    
    return floor_plan

# Path to the input image
image_path = '/image.jpg'

edges, lines = process_image(image_path)
floor_plan = create_floor_plan(edges, lines)

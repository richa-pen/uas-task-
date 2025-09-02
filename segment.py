import cv2
import numpy as np

# Load image
img = cv2.imread('input_image.jpg')
# Convert to HSV for better color segmentation
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define HSV color ranges for ocean (blue) and land (brown/green)
ocean_mask = cv2.inRange(hsv, (90,50,50), (130,255,255))   # Blue
land_mask  = cv2.inRange(hsv, (10,50,50), (40,255,255))    # Brown/Green

# Overlay unique colors for visualization
segmented = img.copy()
segmented[ocean_mask>0] = [255,0,0]      # Ocean: blue
segmented[land_mask>0]  = [34,139,34]    # Land: green

cv2.imwrite('segmented_output.jpg', segmented 

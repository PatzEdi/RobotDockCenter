import os
import cv2
import numpy as np

# NOTE:
    # This script is used for the synthetic world. For the real world images,
    # we will have to manually mark the targets

current_script_path = os.path.dirname(os.path.abspath(__file__))

# We need to get the contour points of green and red targets
def get_contour_points(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the range of green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    # Define the range of red color in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    # Create masks for green and red colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    # Find contours for green and red colors
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Get the contour points for green and red targets
    green_points = []
    red_points = []
    for contour in contours_green:
        for point in contour:
            green_points.append(point[0])
    for contour in contours_red:
        for point in contour:
            red_points.append(point[0])
    return green_points, red_points # Returns two tuples of points

data_dir = os.path.join(current_script_path, "../../godot/data_images")
test_image_path = os.path.join(data_dir,"image1.png")

target_coordinates = get_contour_points(test_image_path)
# Let's print out the coordinates:
print(f"\nTarget 1: {target_coordinates[0]}")
print(f"\nTarget 2: {target_coordinates[1]}")

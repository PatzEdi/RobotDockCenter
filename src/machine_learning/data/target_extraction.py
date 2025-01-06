import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

    # Define the ranges of red color in HSV (need two ranges)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for green and red colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # Create two masks for red and combine them
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2) # Combine the two masks

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

    return green_points, red_points

# Second function that relies on the function above
def get_target_point(target_coordinates):

    # First, we need to get the average of the contour points, so that there
    # is only one point for each target
    # For the first target
    x_sum = 0
    y_sum = 0
    for target_coords in target_coordinates:
        x_sum += target_coords[0]
        y_sum += target_coords[1]

    x_avg = x_sum / len(target_coordinates)
    y_avg = y_sum / len(target_coordinates)

    return x_avg, y_avg

# Here we have a function that separates images based on the targets they have
def get_images_for_each_target(data_dir):
    green_images = []
    red_images = []
    for image in os.listdir(data_dir):
        if image.endswith(".png"):
            image_path = os.path.join(data_dir, image)
            green_points, red_points = get_contour_points(image_path)

            if len(green_points) > 0:
                x_avg_green, y_avg_green = get_target_point(green_points)
                green_images.append((image_path, (x_avg_green, y_avg_green)))
            # We don't do elif here because an image can have both green and
            # red targets
            if len(red_points) > 0:
                x_avg_red, y_avg_red = get_target_point(red_points)
                red_images.append((image_path, (x_avg_red, y_avg_red)))

    return green_images, red_images

data_dir = os.path.join(current_script_path, "../../godot/data_images")
test_image_path = os.path.join(data_dir,"image_999.png")


green_image_data, red_image_data = get_images_for_each_target(data_dir)
print(f"Num green images: {len(green_image_data)}")
print(f"Num red images: {len(red_image_data)}")
# Note: Obviously, since we have one set of images, which is focused on the
# green/first target, there will be more green images than red images. This
# should't be a problem though, as there are still around 3800 images for red,
# when compared to around 4500 images for green

# THE BELOW FUNCTIONS ARE FOR TESTING PURPOSES ONLY:

def plot_image_with_points(image_data):
    image_path = image_data[0]
    # So connecting to that comment above, here we
    # will get the average x and y instead of the target coordinates.
    x_avg, y_avg = image_data[1]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    print(x_avg)
    print(y_avg)
    # Plot the points on the image
    plt.scatter(x_avg, y_avg, color="red")
    plt.show()

# We first extract the contour points via opencv
def iterate_through_images(image_data):
    for image in image_data:
        # Note: 'image' is a tuple, where the first element is the image path
        # and the second element is the contour points
        print(f"Image: {image[0]}")
        plot_image_with_points(image)


test = True
if test:
    iterate_through_images(green_image_data)

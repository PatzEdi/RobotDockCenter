import os
import cv2
import numpy as np
import torch
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

    # Get points and normalize them 0-1
    x_avg = x_sum / len(target_coordinates) / 512
    y_avg = y_sum / len(target_coordinates) / 512

    return x_avg, y_avg

# Here we have a function that separates images based on the targets they have
def get_images_for_each_target(data_dir, n_images_cls=-1):
    #
    #
    # TODO: Add a counter to this function so that we can
    # define how many images we want to load (especially useful for inference
    # and/or splitting the data into training and testing sets)
    #
    #
    green_images = []
    red_images = []
    for image in os.listdir(data_dir):
        # Check if we have the amount of images we want
        if len(green_images) == n_images_cls and len(red_images) == n_images_cls:
            break
        if image.endswith(".png"):
            image_path = os.path.join(data_dir, image)
            green_points, red_points = get_contour_points(image_path)
            # Get the target points for green. Only add the image if we haven't
            # reached out limit for the green class
            if len(green_points) > 0 and len(green_images) != n_images_cls:
                x_avg_green, y_avg_green = get_target_point(green_points)
                x_y_tensor = torch.tensor(
                    [x_avg_green, y_avg_green],
                    dtype=torch.float32
                )
                green_images.append((image_path, x_y_tensor))
            # We don't do elif here because an image can have both green and
            # red targets. Also, same thing here. Only add the image if we
            # haven't reached our limit for the red class.
            if len(red_points) > 0 and len(red_images) != n_images_cls:
                x_avg_red, y_avg_red = get_target_point(red_points)
                x_y_tensor = torch.tensor(
                    [x_avg_red, y_avg_red],
                    dtype=torch.float32
                )
                red_images.append((image_path, x_y_tensor))

    return green_images, red_images

# THE BELOW FUNCTIONS ARE FOR TESTING PURPOSES ONLY:

def plot_image_with_points(image_data):
    image_path = image_data[0]
    # So connecting to that comment above, here we
    # will get the average x and y instead of the target coordinates.
    x_avg, y_avg = image_data[1].tolist()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    print("Coords:")
    print(x_avg*512)
    print(y_avg*512)
    # Plot the points on the image
    plt.scatter(x_avg*512, y_avg*512, color="red")
    plt.show()

# We first extract the contour points via opencv
def iterate_through_images(image_data):
    for image in image_data:
        # Note: 'image' is a tuple, where the first element is the image path
        # and the second element is the contour points
        print(f"Image: {image[0]}")
        plot_image_with_points(image)


def plot_image_contour_surround(image_path):

    points = get_contour_points(image_path)
    green_points_x = [point[0] for point in points[0]]
    green_points_y = [point[1] for point in points[0]]
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    # Plot the points on the image
    plt.scatter(green_points_x, green_points_y, color="red", s=10)
    # Show the image:
    plt.show()

if __name__ == "__main__":


    data_dir = os.path.join(current_script_path, "../../godot/data_images")
    test_image_path = os.path.join(data_dir,"image_999.png")

    # Get the target points. If you want a set amount per class, you can
    # specify the amount with the n_images_cls parameter. If left blank,
    # all images will be loaded
    green_image_data, red_image_data = get_images_for_each_target(data_dir, n_images_cls=12)
    print(f"Num green images: {len(green_image_data)}")
    print(f"Num red images: {len(red_image_data)}")
    print(len(green_image_data))
    print(len(red_image_data))
    # Note: Obviously, since we have one set of images, which is focused on the
    # green/first target, there will be more green images than red images. This
    # should't be a problem though, as there are still around 3800 images for red,
    # when compared to around 4500 images for green
    # Just some testing here...
    test_image_path = green_image_data[11][0]
    
    # Our reference image:
    contour_image = cv2.imread(test_image_path)
    average_image = contour_image.copy() # Create a copy
    
    points = get_contour_points(test_image_path)
    
    green_points_x = [point[0] for point in points[0]]
    green_points_y = [point[1] for point in points[0]]

    # Plot the green points using cv2.circle
    for i in range(len(green_points_x)):
        cv2.circle(contour_image, (green_points_x[i], green_points_y[i]), 2, (0, 0, 255), -1)
    
    x_avg, y_avg = green_image_data[11][1].tolist()
    cv2.circle(average_image, (int(x_avg*512), int(y_avg*512)), 5, (0, 0, 255), -1)
    
    # Add text to the images:
    # For the average_image, we put "After", on the contour we but "Before"
    cv2.putText(contour_image, "Before", (220, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(average_image, "After", (220, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    # Save the images:
    cv2.imwrite(os.path.expanduser("~/Downloads/contour_image.png"), contour_image)
    cv2.imwrite(os.path.expanduser("~/Downloads/average_image.png"), average_image)
    
    if False:
        iterate_through_images(green_image_data)

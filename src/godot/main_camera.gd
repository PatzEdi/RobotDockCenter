extends Camera3D

# We use the DockIndicator (Red Dot) as a reference to the dock itself.
@onready var DockIndicator : MeshInstance3D = get_parent().get_parent().get_parent().get_node("meshes/Dock/DockIndicator")

# Let's define some variables for the reverse line.
var reverse_line_distance_from_dock = 2 # distance between the reverse line and the dock.

# Let's define some hyperparameters for the image gathering:
var save_images = true # Determines whether or not to save the images in the current run.
var num_images_per_class = 300 # To ensure that we have an equal number of images for each class, we will set a fixed amount instead of just using random values for camera movements.
var num_classes = 4 # There are three classes as of now (a fourth will be added later): center, left, and right
var num_classes_counter = 0

var at_rline_point_extra_image_num = (num_images_per_class)*(num_classes-2) # These are extra images to take of just images at the reverse line. This is in order for the data set to be more balanced in terms of the direction value -1, 0, or 1 specifically.
var surpassed_rline_extra_image_num = (num_images_per_class)*(num_classes-2) # Same thing as above but for surpassed images, aka images either too far close to the Dock that aren't centered, or images that are at the reverse line point but are too left or too right.

# This counter will count the amount of images we have taken. It will be used to assign image names e.g. image_1.png, image_2.png, etc.
var global_image_counter = 0

# This below is to determine whether or not the images being gathered will face directly at the reverse line point or not. This will be changed by the code itself using counter variables. Half the images true, half of them false
var perfectly_centered = true

# This below is to save the image lines to then put in a file.
var image_data_lines = []

var reverse_line_pos # This is the reverse line position that image functions from the first phase will be using (first phase is the one pre-reverse line point being reached)
# Called when the node enters the scene tree for the first time.
func _ready():
	await get_tree().process_frame # We wait for the frames to render as an extra precaution to avoid null/black viewport textures.
	
	# We set a title for the elements in the data file in csv format:
	image_data_lines.append("Image,Type,Direction,Distance_rline,Rotation_value")
	
	# Let's print out some information based on the hyperparameters defined above:
	if (save_images):
		print("Save images is true, images will be saved")
	else:
		print("SAVE IMAGES IS FALSE, IMAGES WILL NOT BE SAVED")
	
	# Let's center the camera so that it is not too high but also not too low:
	self.transform.origin.x = DockIndicator.global_position.x + reverse_line_distance_from_dock # We start at the reverse line. This is important because each function that moves the camera then comes back to this position after taking the picture.
	self.transform.origin.y = DockIndicator.global_position.y + .3 # .3 is chosen has the level of the y axis for the camera. It aims it more down toward the reverse line rather than leveled, which would be unrealistic on the robot at hand.
	self.transform.origin.z = DockIndicator.global_position.z
	
	reverse_line_pos = self.transform.origin

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta):
	# We make half of the images facing directly, and half of the images not facing directly.
	if (global_image_counter >= (num_images_per_class/2)+(num_images_per_class*num_classes_counter)):
		perfectly_centered = false # We need to fix this next, possibly by updating a counter variable in the if statements, and replacing the num_classes variable here with that counter variable
	else:
		perfectly_centered = true
	
	
	# NOTE: if statements regarding the counter variables are vital here. We can't just add all the image gathering functions here, because otherwise Godot will execute all the function calls PER FRAME, which is not possible and will cause failure.
	
	if (global_image_counter < num_images_per_class):
		get_center_image() # Get's a single image in the center line.
	elif (global_image_counter < num_images_per_class*2):
		if (num_images_per_class==global_image_counter): # This is to forcefully make the last image of the previous loop perfectly centered.
			perfectly_centered = true
		num_classes_counter = 1
		get_left_image() # Get's a single image to the left of the center line.
	elif (global_image_counter < num_images_per_class*3):
		if (num_images_per_class*2==global_image_counter):
			perfectly_centered = true
		num_classes_counter = 2
		get_right_image() # Get's a single image to the right of the center line.
	elif (global_image_counter < (num_images_per_class*4)+at_rline_point_extra_image_num):
		if (num_images_per_class*3==global_image_counter):
			perfectly_centered = true
		num_classes_counter = 4 # We change this due to the extra images for this class added to make the dataset more equal.
		get_reverse_line_point_image()
	# NOTE: To add another class with extra images like the one above:
	"""
	1. Add the extra image amount to what global_image_counter should be less to.
	2. Change the num_classes counter so that the amount of centered and rotated images is equal, even when taking into account the extra images.
	3. Add the extra image amount to the if statement below, to deterime when the program should stop based on the amount of images that should be taken.
	"""
	if (global_image_counter == (num_images_per_class * num_classes)+at_rline_point_extra_image_num):
		if (save_images):
			print("\nWriting image data lines to file...")
			write_lines_to_file()
		# Data gathering is finished at this point and it is safe to exit the process.
		print("\nGathering of data has finished.")

		get_tree().quit() # terminate the data gathering.

func get_center_image():
	# Put your code here to move the camera:
	var random_x = randf_range(.5,7)
	
	self.transform.origin.x += random_x
	
	var rotation_value = get_rotation_value()
	
	await get_tree().process_frame # We wait for the frame to render
	
	var distance_from_reverse_line = self.transform.origin.distance_to(reverse_line_pos)
	
	var img = get_viewport().get_texture().get_image()
	
	global_image_counter += 1
	
	# Save the image and append to the image_data_lines list.
	if (save_images):
		img.save_png("res://data_images/image_" + str(global_image_counter) + ".png")
		image_data_lines.append("image_" + str(global_image_counter) + ".png,center,1," + str(distance_from_reverse_line) + "," + str(rotation_value))
	
	self.transform.origin = reverse_line_pos # Then, we move back to the original position, which is the reverse point line.

func get_left_image():
	# Put your code here to move the camera:
	# We now not only need a random number for the x (vertical), but we also need a random value for the z (horizontal)
	var random_x = randf_range(.5,7)
	var random_z = randf_range(.5,4)# For the left, we need to generate positive numbers
	
	self.transform.origin.x += random_x
	self.transform.origin.z += random_z
	
	var rotation_value = get_rotation_value(25) # We get the rotation value needed to face the reverse point indicator
	
	await get_tree().process_frame # We wait for the frame to render
	
	var distance_from_reverse_line = self.transform.origin.distance_to(reverse_line_pos)
	
	var img = get_viewport().get_texture().get_image()
	
	global_image_counter += 1
	
	# Put your code here to save the image:
	if (save_images):
		img.save_png("res://data_images/image_" + str(global_image_counter) + ".png")
		image_data_lines.append("image_" + str(global_image_counter) + ".png,left,1," + str(distance_from_reverse_line) + "," + str(rotation_value))
	
	self.transform.origin = reverse_line_pos # We move the camera back to its original position, which is the reverse point line.

func get_right_image():
	# Put your code here to move the camera:
	# We now not only need a random number for the x (vertical), but we also need a random value for the z (horizontal)
	var random_x = randf_range(.5,7)
	var random_z = randf_range(-4,-.5)# For the right, we need to generate positive numbers
	
	self.transform.origin.x += random_x
	self.transform.origin.z += random_z
	
	var rotation_value = get_rotation_value(25) # We get the rotation value needed to face the reverse point indicator
	
	await get_tree().process_frame # We wait for the frame to render
	
	var distance_from_reverse_line = self.transform.origin.distance_to(reverse_line_pos)
	
	var img = get_viewport().get_texture().get_image()
	
	global_image_counter += 1
	
	# Put your code here to save the image:
	if (save_images):
		img.save_png("res://data_images/image_" + str(global_image_counter) + ".png")
		image_data_lines.append("image_" + str(global_image_counter) + ".png,right,1," + str(distance_from_reverse_line) + "," + str(rotation_value))
	
	self.transform.origin = reverse_line_pos # We move the camera back to its original position

# This will be done soon as well, it is to get images at the reverse line point. Instead of looking at the reverse point line, we look at the DockIndicator, and thus the rotation value will be according to the Dock indicator.
# For these images, there will be some threshold in terms of distance. Perhaps +/= the reverse_line_distance_from_dock
func get_reverse_line_point_image():
	
	var random_x = randf_range(-.5,.5) # We set a small threshold so as to not make the reverse line point exact/very hard to reach.
	self.transform.origin.x += random_x # We move the camera
	
	# We now look at the dock:
	var rotation_value = get_rotation_value(20,true)
	
	await get_tree().process_frame
	
	var distance_from_reverse_line = self.transform.origin.distance_to(reverse_line_pos)
	# The reverse line point is essentially a rectangle with some threshold to it. It isn't a perfect point.
	var img = get_viewport().get_texture().get_image()
	
	global_image_counter += 1
	# Put your code here to save the image:
	if (save_images):
		img.save_png("res://data_images/image_" + str(global_image_counter) + ".png")
		# We apply the distance value even if it isn't zero, so that the model learns that there is some threshold.
		image_data_lines.append("image_" + str(global_image_counter) + ".png,at_rline,0," + str(distance_from_reverse_line)+ "," + str(rotation_value))

	self.transform.origin = reverse_line_pos
# These images will be images that have surpassed the reverse line point, or that they are too left or right to be considered to be in the reverse line point.
func get_surpassed_images():
	# Put your code here:
	
	
	global_image_counter += 1
	
	# Put your code here to save the image:
	if (save_images):
		pass

# This is for the rotation value:
func get_rotation_value(degree_range=30, look_at_dock=false):
	var rotation_value = 0
	if (look_at_dock):
		self.look_at(Vector3(DockIndicator.global_position.x, DockIndicator.global_position.y, DockIndicator.global_position.z))
	else:
		# We look at the reverse line point.
		self.look_at(Vector3(DockIndicator.global_position.x + reverse_line_distance_from_dock, DockIndicator.global_position.y, DockIndicator.global_position.z))
	
	# After that, we add a random rotation to it if perfectly_centered is false. perfectly centered will most likely be a global variable, rater than a parameter of these functions.
	if (!perfectly_centered):
		# The random rotation still needs to be thought out. Instead of a random number within a fixed range, we should most likely make such a range adaptive according to the distance from the reverse point and the camera. The farther away, the greater the range, and vice versa. This is because as we get closer, the ability to fit in the entire within a fixed range dock gets smaller. (Verify this? Perhaps we want it to learn to move even when only some of the dock can be seen)
		# For now, though, we use a fixed range.
		# The degree_range parameter is in plus/minus. In other words it is the max amount that the random rotation can go away from 0, either positive or negative. This IS in degrees.
		rotation_value = randf_range(-degree_range, degree_range) # The random rotation value, which will also be used as a data point for the image being taken.
		self.rotation.y += deg_to_rad(rotation_value) # IMPORTANT: We need to convert the degrees generated into radians, as that is how Godot stores the rotation values.
		# Note: Positive rotation_value = rotate left. Negative rotation value = rotate right.
	return rotation_value



# TEST FUNCTIONS (The below functions are used for quick tests):

func test_rotation_difference(): # Used to see if rotation is retrievable and what values it corresponds to (initial testing purposes):
	var rotation_value = 0
	# We look at the reverse line point.
	self.look_at(Vector3(DockIndicator.global_position.x + reverse_line_distance_from_dock, DockIndicator.global_position.y, DockIndicator.global_position.z))
	# After that, we add a random rotation to it if perfectly_centered is false. perfectly centered will most likely be a global variable, rater than a parameter of these functions.
	if (!perfectly_centered):
		# The random rotation still needs to be thought out. Instead of a random number within a fixed range, we should most likely make such a range adaptive according to the distance from the reverse point and the camera. The farther away, the greater the range, and vice versa. This is because as we get closer, the ability to fit in the entire within a fixed range dock gets smaller. (Verify this? Perhaps we want it to learn to move even when only some of the dock can be seen)
		# For now, though, we use a fixed range.
		var degree_range = 30 # This is in plus/minus. In other words it is the max amount that the random rotation can go away from 0, either positive or negative. This IS in degrees.
		rotation_value = randi_range(-degree_range, degree_range) # The random rotation value, which will also be used as a data point for the image being taken.
		self.rotation.y += deg_to_rad(rotation_value) # IMPORTANT: We need to convert the degrees generated into radians, as that is how Godot stores the rotation values.
		# Note: Positive rotation_value = rotate left. Negative rotation value = rotate right.
	return rotation_value

func test_positional_movement_x():
	# We store the original position (at the reverse line) in a variable, so that after we do the movement, we can go back to this position:
	var original_pos = self.transform.origin
	var random_x = randi_range(1,7) # For now, we do 1-7 in order to avoid look_at debug errors. If 0, it would be directly above.
	# Put your code here, to move the camera:
	self.transform.origin.x += random_x
	# We get the rotation value
	var rotation_value = test_rotation_difference()
	
	await get_tree().process_frame # This is vital, and is needed in order to render the viewport before capturing the image.
	
	var distance_from_reverse_line = self.transform.origin.distance_to(original_pos)
	
	var img = get_viewport().get_texture().get_image()
	
	# Save the image and append to the image_data_lines list.
	if (save_images):
		img.save_png("res://data_images/image_" + str(global_image_counter) + ".png")
		image_data_lines.append("image_" + str(global_image_counter) + ", Distance rline: " + str(distance_from_reverse_line) + ", random_x: " + str(random_x) + ", rotation_value: " + str(rotation_value))
	
	# Then, we move back to the original position. We will do this for all three of the get_center, get_left, and get_right funcs.
	self.transform.origin = original_pos

# Used to write the contents of image_data_lines to a file:
func write_lines_to_file():
	var file = FileAccess.open("res://data_text/image_info_data.txt", FileAccess.WRITE)
	for element in image_data_lines:
		file.store_line(element)
	file.close()


# NOTE: # Something to be aware of on line 142 (This is related to the refactoring/repositioning of things, said down below at line 173): When this random_x is 0, it means the camera is at the reverse_line point. Since the camera is facing the reverse line point, the picture shows as slanted, but with a rotation value is messed up. To fix this, the rotation value would have to be changed according to the distance value. If the distance value is 0, that is when the camera should use the DockIndicator as a reference point rather than the reverse line point, as that is when the next phase of the autodock starts.
# Ah, I found the error of the look_at. It was about the random_x. When it would be 0, it would be right on top of the reverse line point, causing the error.

# So what is the next thing to do? Restructure the code to use the code in theese test functions, but make it so that it differentiates between when the camera is directly on the reverse line point and when it is above.
# Essentially just figure out how everything should be structured in terms of ensuring that the data will be equal for all of the image classes.

extends Camera3D

@onready var DockIndicator : MeshInstance3D = get_parent().get_parent().get_parent().get_parent().get_node("meshes/Dock/DockIndicator")
var rline_pos # This position represents the very center of the rline.
var rline_pos_w_threshold # This is used to store the rline position with the threshold below.
var rline_threshold = .5 # This is the threhold to determine the "thickness" of the rline up/down. Multiply this value by two to get the total "thickness".
# Some rline params:
var distance_rline_dock = 2 # The perpendicular distance from the center of the reverse line and the dock
# Camera step positioning and movement params:
var starting_distance = 7 # The starting distance from the rline_pos_w_threshold
var starting_distance_z = 4 # The starting distance from the cline every new row.

var x_axis_step_amount = .5
var z_axis_step_amount = .5

# Camera rotation step movements.
var rotation_step_amount = 2
var rotation_range = 25 # Rotation range in degrees (-range, range)

var total_x_step_points = ((starting_distance)/x_axis_step_amount) + 1
var total_z_step_points_per_row = (starting_distance_z*2/z_axis_step_amount) + 1
var total_step_points = total_x_step_points * total_z_step_points_per_row

var total_rotations_per_step_point = ((rotation_range*2)/rotation_step_amount) + 1
var total_num_images = total_step_points * total_rotations_per_step_point

"""
Now we have the variables for overall variable counting, such as when to move to the next row of step points for the data gathering.
"""
var z_steps_counter = 0 # This will be reset every time the camera has reached the end of the row.
var x_steps_counter = 0 # This will not be reset but rather added onto each time the end of a row is reached.
var rotation_steps_counter = 0 # This will be reset for each z-step.

var model = 1 # select either 1 or 2 to determine whether or not images & their rotation values will be focused on the rline_pos (model1) or the DockIndicator itself (model2)
var target # This will either by rline_pos or DockIndicator.transform.origin, based on the model selected above (1 or 2).

# boolean value to determine whether or not the camera should save its viewport image to the disk or not.
var save_images = false
# Used to determine the name of the text file to use depending on the model chosen above
var data_csv_file_name = "image_info_data_model" + str(model)
# This below is used to determine where to save the images, based on the model chosen.
var data_images_folder = "data_images"
# This is the list to save the lines for the text data:
var image_data_lines = []

# image counter for the image names:
var global_image_counter = 0
# Called when the node enters the scene tree for the first time.
func _ready():
	image_data_lines.append("Image,RotationValue,DistanceCline") # We first append the headers of the csv data file
	# We notify/make it clear if the data will be overwritten/images will be saved
	if (save_images):
		print("Save images is true, images will be saved")
	else:
		print("SAVE IMAGES IS FALSE, IMAGES WILL NOT BE SAVED")
	
	print("\nZ-Step Points per row: " + str(total_z_step_points_per_row)) # We multiply by two because the total amount of distance for the z goes from 4 to the left, to 4 to the right.
	print("Total X-Step Points: " + str(total_x_step_points))
	
	print("Total rotation step points: " + str(total_rotations_per_step_point))
	var calculation_string = str(total_z_step_points_per_row) + "*" + str(total_x_step_points) + "*" + str(total_rotations_per_step_point)
	
	# Then, we print out the total number of images that we will have:
	print("\nTotal number of images: " + calculation_string + " = " + str(total_num_images) + "\n")
	
	# Let's set the correct origin values for the rline, so that we can come back to it for reference if needed later on.
	rline_pos = DockIndicator.transform.origin
	rline_pos.x += distance_rline_dock
	
	rline_pos_w_threshold = rline_pos
	rline_pos_w_threshold.x += rline_threshold # We add the threshold num as that is the farthest 'upper' threshold that defines the area defined by the rline. We also add it here so that it is already accounted for.
	# Let's move our camera to where the the very end threshold of the rline is, for proper positioning.
	self.transform.origin = rline_pos_w_threshold
	
	# Here we decide what to change based on the model selected, which includes the images path to save the images for the specific model chosen, and the target to look at
	if (model == 1):
		print("Model 1 is chosen, images will be focalized on rline_pos\n")
		target = rline_pos
	else:
		print("Model 2 is chosen, images will be focalized on DockIndicator\n")
		target = DockIndicator.transform.origin
		data_images_folder += str(2)
		
	"""
	Once we make sure that the camera is ready to move from a perfectly centered position with the Dock, we can move the camera to the starting point of the data gathering process.
	"""
	self.transform.origin.x += starting_distance
	self.transform.origin.z = DockIndicator.transform.origin.z - starting_distance_z
	# Here we also need to center and rotate in terms of rotation like in the z_step and x_step functions
	self.look_at(target)
	# Here we turn.
	self.rotation.y += deg_to_rad(rotation_range)

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta):
	await get_tree().process_frame # Let's wait for the frame to process before gathering the image.

	# Below we get the rotation values
	# START DEBUG ROTATIONS
	var temp_rotation = self.rotation.y
	self.look_at(target)
	var rotation_value = rad_to_deg(temp_rotation-self.rotation.y)
	self.rotation.y = temp_rotation
	# END DEBUG ROTATIONS
	
	# Below we get the distance_cline values:
	var distance_cline = self.transform.origin.z-DockIndicator.transform.origin.z
	
	# Here we put the code to get the image from the viewport: 
	if save_images:
		var img = get_viewport().get_texture().get_image()
		img.save_png("res://" + data_images_folder + "/image_" + str(global_image_counter+1) + ".png")
		image_data_lines.append("image_" + str(global_image_counter+1) + ".png," + str(rotation_value) + ","+str(distance_cline))

	if (rotation_steps_counter == total_rotations_per_step_point-1):
		rotation_steps_counter = 0
		z_step()
	else:
		rot_step()
	if (z_steps_counter == total_z_step_points_per_row):
		z_steps_counter = 0
		x_step()
	# If we are done gathering the images, we exit.
	if (x_steps_counter == total_x_step_points):
		
		if (save_images):
			write_lines_to_file()
			
		get_tree().quit()
	
	# We add one to the image counter every frame.
	global_image_counter += 1
	
# Below we put our functions:
func z_step():
	# Here we simple move the camera left by the z-step amount:
	self.transform.origin.z += z_axis_step_amount
	# Here we reset the rotation as well, according to the rotation range:
	self.look_at(target)
	self.rotation.y += deg_to_rad(rotation_range)
	z_steps_counter += 1
func x_step():
	# Here we simple move the camera torwards the rline by the x-step amount and reset the distance cline for the z-axis as well.
	self.transform.origin.x -= x_axis_step_amount
	self.transform.origin.z = DockIndicator.transform.origin.z - starting_distance_z
	# For each x_step we do, we do what the z_step() method does as well, which is center the camera and rotate based on the range.
	self.look_at(target)
	# Here we turn.
	self.rotation.y += deg_to_rad(rotation_range)
	x_steps_counter += 1
func rot_step():
	"""
		As we move about the zaxis and xaxis, the angles for each step point change.
		
		For example, as we move along the z axis to the right, the angle becomes more acute to face toward the reverse line point.
		
		To solve this, we can, from any step point, look at the rline point, and add the rotation angle range to it, and then start stepping from there.
	"""
	self.rotation.y -= deg_to_rad(rotation_step_amount)
	
	rotation_steps_counter += 1

# This below is to write the data to the file we will use for training:
func write_lines_to_file():
	var file = FileAccess.open("res://data_text/" + data_csv_file_name + ".csv", FileAccess.WRITE)
	for element in image_data_lines:
		file.store_line(element)
	file.close()

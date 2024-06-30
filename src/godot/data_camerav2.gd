extends Camera3D

@onready var DockIndicator : MeshInstance3D = get_parent().get_parent().get_parent().get_parent().get_node("meshes/Dock/DockIndicator")
var rline_pos # This position represents the very center of the rline.
var rline_threshold = .5 # This is the threhold to determine the "thickness" of the rline up/down. Multiply this value by two to get the total "thickness".
# Some rline params:
var distance_rline_dock = 2 # The perpendicular distance from the center of the reverse line and the dock
# Camera step positioning and movement params:
var starting_distance = 7 # The starting distance from the rline_pos
var starting_distance_z = 4 # The starting distance from the cline every new row.

var x_axis_step_amount = .5
var z_axis_step_amount = .5

# Camera rotation step movements.
var rotation_step_amount = 2
var rotation_range = 20 # Rotation range in degrees (-range, range)

var total_x_step_points = (starting_distance)/x_axis_step_amount
var total_z_step_points_per_row = starting_distance_z*2/z_axis_step_amount
var total_step_points = total_x_step_points * total_z_step_points_per_row

var total_rotations_per_step_point = (rotation_range*2)/rotation_step_amount
var total_num_images = total_step_points * total_rotations_per_step_point

"""
Now we have the variables for overall variable counting, such as when to move to the next row of step points for the data gathering.
"""
var z_steps_counter = 0 # This will be reset every time the camera has reached the end of the row.
var x_steps_counter = 0 # This will not be reset but rather added onto each time the end of a row is reached.
var rotation_steps_counter = 0 # This will be reset for each z-step.


# boolean value to determine whether or not the camera should save its viewport image to the disk or not.
var save_images = false
# This is the list to save the lines for the text data:
var image_data_lines = []

# image counter for the image names:
var global_image_counter = 0
# Called when the node enters the scene tree for the first time.
func _ready():
	# First we notify/make it clear if the data will be overwritten/images will be saved
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
	
	# Let's set the correct origin values for the rline, as that we can come back to it for reference if needed later on.
	rline_pos = DockIndicator.transform.origin
	rline_pos.x += distance_rline_dock + rline_threshold # We add the threshold num as that is the farthest 'upper' threshold that defines the area defined by the rline. We also add it here so that it is already accounted for.
	# Let's move our camera to where the the very end threshold of the rline is, for proper positioning.
	self.transform.origin = rline_pos
	
	"""
	Once we make sure that the camera is ready to move from a perfectly centered position with the Dock, we can move the camera to the starting point of the data gathering process.
	We don't need to do this initial positioning in the _ready() function besides the x-axis (starting_distance), but rather have it just normally, as it goes by a 0 * step amount and such.
	"""
	# The only initial positioning we need to do is the x axis. We add the starting distance on top of the threshold accounted for before.
	self.transform.origin.x += starting_distance
	
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	await get_tree().process_frame # Let's wait for the frame to process before gathering the image.
	# Here we put the code to get the image from the veiwport: 
	if save_images:
		var img = get_viewport().get_texture().get_image()
		img.save_png("res://data_images/image_" + str(global_image_counter) + ".png")
		image_data_lines.append("image_" + str(global_image_counter+1) + ".png,center,")#+","+ str(distance_from_reverse_line) + "," + str(rotation_value))
	# Here we put if statements to check e.g. if we need to go to the next row or something, and then call the functions under where it says "Below we put our functions:":
	if (rotation_steps_counter == total_rotations_per_step_point-1):
		# Here, once the last rotation is finished, we reset the rotation counter:
		rotation_steps_counter = 0
		z_step() # Then we move left.
		# And then after this we reset the camera for the next set of rotation images by looking at the rline point, and then turning right/left by the rotation range.
	# If this is the case, we move to the next row:
	elif (z_steps_counter == total_z_step_points_per_row-1):
		z_steps_counter = 0
		x_step()
	# If we are done gathering the images, we exit.
	if (global_image_counter == total_num_images-1):
		get_tree().quit()
	# We add one to the image counter every frame.
	global_image_counter += 1
	
# Below we put our functions:
func z_step():
	# Here we simple move the camera left by the z-step amount:
	self.transform.origin.z += z_axis_step_amount
	z_steps_counter += 1
func x_step():
	# Here we simple move the camera torwards the rline by the x-step amount and reset the distance cline for the z-axis as well.
	self.transform.origin.x += x_axis_step_amount * x_steps_counter
	self.transform.origin.z = DockIndicator.transform.origin.z + starting_distance_z
	x_steps_counter += 1
func rot_step():
	"""
	We have a bit of a problem:
		
		As we move about the zaxis and xaxis, the angles for each step point change.
		
		For example, as we move along the z axis to the right, the angle becomes more acute to face toward the reverse line point.
		
		To solve this, we can, from any step point, look at the rline point, and add the rotation angle range to it, and then start stepping from there.
	"""
	rotation_steps_counter += 1


# This below is to write the data to the file we will use for training:
func write_lines_to_file():
	var file = FileAccess.open("res://data_text/image_info_data.csv", FileAccess.WRITE)
	for element in image_data_lines:
		file.store_line(element)
	file.close()

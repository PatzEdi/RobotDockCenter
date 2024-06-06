extends Camera3D

# We use the DockIndicator (Red Dot) as a reference to the dock itself.
@onready var DockIndicator : MeshInstance3D = get_parent().get_node("meshes/Dock/DockIndicator")

# Let's define some variables for the reverse line.
var reverse_line_distance_from_dock = 2 # distance between the reverse line and the dock.

# Let's define some hyperparameters for the image gathering:
var save_images = false # Determines whether or not to save the images in the current run.
var num_images_per_class = 1000 # To ensure that we have an equal number of images for each class, we will set a fixed amount instead of just using random values for camera movements.
var num_classes = 3 # There are three classes as of now (a fourth will be added later): center, left, and right

# This counter will count the amount of images we have taken. It will be used to assign image names e.g. image_1.png, image_2.png, etc.
var global_image_counter = 0

# Called when the node enters the scene tree for the first time.
func _ready():
	# Let's print out some information based on the hyperparameters defined above:
	if (save_images):
		print("Save images is true, images will be saved")
	else:
		print("SAVE IMAGES IS FALSE, IMAGES WILL NOT BE SAVED")
	
	# Let's center the camera so that it is not too high but also not too low:
	self.transform.origin.x = DockIndicator.global_position.x + 5 # The "+2" doesn't change anything. Feel free to remove it.
	self.transform.origin.y = DockIndicator.global_position.y + .3 # .3 is chosen has the level of the y axis for the camera. It aims it more down toward the reverse line rather than leveled, which would be unrealistic on the robot at hand.
	self.transform.origin.z = DockIndicator.global_position.z


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta):
	# Put your code here, to move the camera (e.g functions):
	get_center_image() # Get's a single image in the center line.
	
	get_left_image() # Get's a single image to the left of the center line.
	
	get_right_image() # Get's a single image to the right of the center line.
	
	if (global_image_counter == num_images_per_class * num_classes):
		# Data gathering is finished at this point and it is safe to exit the process.
		print("\nGathering of data has finished.")
		get_tree().quit() # terminate the data gathering.
	
# PARAMS for the functions below:
# 1. perfectly_centered deterimes whether or not the pictures taken are directly facing the reverse line point (true). If this is toggled off (false), the pictures taken will not be facing directly, and thus there will be a rotation value other than 0.

func get_center_image(perfectly_centered=false):
	# Put your code here to move the camera:
	
	
	if (perfectly_centered):
		# Look at the reverse line position. Note: We don't want to edit the y axis values for this
		self.look_at(Vector3(DockIndicator.global_position.x + reverse_line_distance_from_dock, DockIndicator.global_position.y, DockIndicator.global_position.z))
	
	global_image_counter += 1
	
	# Put your code here to save the image:
	if (save_images):
		pass
	


func get_left_image(perfectly_centered=false):
	# Put your code here to move the camera:
	
	
	if (perfectly_centered):
		# Look at the reverse line position.
		self.look_at(Vector3(DockIndicator.global_position.x + reverse_line_distance_from_dock, DockIndicator.global_position.y, DockIndicator.global_position.z))
	
	global_image_counter += 1
	
	# Put your code here to save the image:
	if (save_images):
		pass

func get_right_image(perfectly_centered=false):
	# Put your code here to move the camera:
	
	
	if (perfectly_centered):
		# Look at the reverse line position.
		self.look_at(Vector3(DockIndicator.global_position.x + reverse_line_distance_from_dock, DockIndicator.global_position.y, DockIndicator.global_position.z))
	
	global_image_counter += 1
	
	# Put your code here to save the image:
	if (save_images):
		pass


# This will be configured later. It is used to get the images that have passed the reverse line. This is useful if the robot is placed passed the reverse line manually. It will consist of going backward rather than forward.
func get_surpassed_images(perfectly_centered=false):
	# Put your code here:
	
	
	if (perfectly_centered):
		# Look at the reverse line position.
		self.look_at(Vector3(DockIndicator.global_position.x + reverse_line_distance_from_dock, DockIndicator.global_position.y, DockIndicator.global_position.z))
	
	global_image_counter += 1
	
	# Put your code here to save the image:
	if (save_images):
		pass

extends Camera3D

@onready var DockIndicator : MeshInstance3D = get_parent().get_parent().get_parent().get_parent().get_node("meshes/Dock/DockIndicator")

# Camera step positioning and movement params:
var starting_distance=7 # The starting distance from the rline at the start.
var starting_distance_z = 4 # The starting distance from the cline every new row.

var x_axis_step_amount = .5
var z_axis_step_amount = .5

# Camera rotation step movements.
var rotation_step_amount = 2
var rotation_range = 25 # Rotation range in degrees (-range, range)

var x_end_point_num = .5 # This is the threhold to determine when to stop the data gathering, and therefore also where the reverse line starts.

var total_x_step_points = (starting_distance-x_end_point_num)/x_axis_step_amount
var total_z_step_points_per_row = starting_distance_z*2/z_axis_step_amount
var total_step_points = total_x_step_points * total_z_step_points_per_row

var total_rotations_per_step_point = (rotation_range*2)/rotation_step_amount
var total_num_images = total_step_points * total_rotations_per_step_point

"""
Now we have the variables for overall variable counting, such as when to move to the next row of step points for the data gathering.
"""
var z_steps_counter = 0 # This will be reset every time the camera has reached the end of the row.
var x_steps_counter = 0 # This will not be reset but rather added onto each time the end of a row is reached.

# Called when the node enters the scene tree for the first time.
func _ready():
	print("\nZ-Step Points per row: " + str(total_z_step_points_per_row)) # We multiply by two because the total amount of distance for the z goes from 4 to the left, to 4 to the right.
	print("Total X-Step Points: " + str(total_x_step_points))
	
	print("Total rotation step points: " + str(total_rotations_per_step_point))
	var calculation_string = str(total_z_step_points_per_row) + "*" + str(total_x_step_points) + "*" + str(total_rotations_per_step_point)
	
	# Then, we print out the total number of images that we will have:
	print("\nTotal number of images: " + calculation_string + " = " + str(total_num_images) + "\n")


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

extends Camera3D

@onready var DockIndicator : MeshInstance3D = get_parent().get_parent().get_parent().get_parent().get_node("meshes/Dock/DockIndicator")
@onready var robot_body : MeshInstance3D = get_parent().get_parent().get_parent().get_parent().get_node("meshes/robot/body")

var http_request
# Called when the node enters the scene tree for the first time.
func _ready():
	
	await get_tree().process_frame
	http_request = HTTPRequest.new()
	http_request.request_completed.connect(_on_request_completed)
	self.add_child(http_request)
	# Let's set a random starting point for the robot:
	var random_z = randf_range(-4,4)
#
#
#	# We transform both the robot body and the camera:
	robot_body.transform.origin.z += random_z
	self.transform.origin.z += random_z
	rotate_randomly()
	await get_tree().process_frame
	send_image()
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta):
	pass
	
# Networking functions:
func send_image():
	var img = get_viewport().get_texture().get_image() # Get the img object.
	await get_tree().process_frame # Wait for the fram to process.
	#img.save_png("res://frames_testing/frame.png")
	var png_data = img.save_png_to_buffer()
	var base64_image_data = Marshalls.raw_to_base64(png_data) # Convert the image into base64 encoding.
	
	var custom_headers = ["Content-Type: application/json"]
	var request_data = JSON.stringify({"image": base64_image_data}) # Then, we convert it into JSON format. This will be the payload.

	http_request.request("http://127.0.0.1:5000/predict",custom_headers, HTTPClient.METHOD_POST, request_data) # Send the image data to the server.

func _on_request_completed(_result, _response_code, _headers, body):
	var json = JSON.parse_string(body.get_string_from_utf8())
	var predicted_values = Array(json["predicted_values"]) # Extract the predicted values
	print("Predicted values: ", predicted_values)
	
	# We put the code to move the robot here:
	parse_predictions_and_move(predicted_values[0], predicted_values[1])
	
	send_image()

# This is used for the initial placement of the robot
func rotate_randomly(degree_range=25):
	var rotation_value = 0

	# We look at the reverse line point.
	self.look_at(Vector3(DockIndicator.global_position.x, DockIndicator.global_position.y, DockIndicator.global_position.z))
	# The degree_range parameter is in plus/minus. In other words it is the max amount that the random rotation can go away from 0, either positive or negative. This IS in degrees.
	rotation_value = randf_range(-degree_range, degree_range) # The random rotation value, which will also be used as a data point for the image being taken.
	self.rotation.y += deg_to_rad(rotation_value) # IMPORTANT: We need to convert the degrees generated into radians, as that is how Godot stores the rotation values.
	# We also rotate the robot body:
	robot_body.rotation.y += deg_to_rad(rotation_value)

# Functions to move the robot:
func parse_predictions_and_move(rotation_value, distance):
	await get_tree().process_frame
	var bot_movement_amount = .5
	var bot_rotation_amount = .3
	
	if (rotation_value < -1):
		self.rotation.y += deg_to_rad(bot_rotation_amount)
		robot_body.rotation.y += deg_to_rad(bot_rotation_amount)
	elif (rotation_value > 1):
		self.rotation.y += deg_to_rad(-bot_rotation_amount)
		robot_body.rotation.y += deg_to_rad(-bot_rotation_amount)
	else:
		var fps = Engine.get_frames_per_second()
		var delta = 1.0 / fps
		
		# Here we would put an if else statement regarding whether or not to move forward or backward.
		robot_body.global_transform.origin -= global_transform.basis.z * bot_movement_amount * delta
		self.global_transform.origin -= global_transform.basis.z * bot_movement_amount * delta
		
	# positive rotations = turn right, negative rotations = turn left.
#	if (rotation_value < -1):
#		#robot_body.rotation.y += deg_to_rad(bot_rotation_amount)
#		self.rotation.y += deg_to_rad(bot_rotation_amount)
#	elif (rotation_value > 1):
#		#robot_body.rotation.y += deg_to_rad(-bot_rotation_amount)
#		self.rotation.y += deg_to_rad(-bot_rotation_amount)
#	else:
#		var fps = Engine.get_frames_per_second()
#		var delta = 1.0 / fps
#		#robot_body.global_transform.origin -= robot_body.global_transform.basis.z * bot_movement_amount * delta
#		self.global_transform.origin -= global_transform.basis.z * bot_movement_amount * delta

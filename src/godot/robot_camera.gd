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
    var random_z = randf_range(-2,2)
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

    http_request.request("http://127.0.0.1:5000/get_action",custom_headers, HTTPClient.METHOD_POST, request_data) # Send the image data to the server.

func _on_request_completed(_result, _response_code, _headers, body):
    var json = JSON.parse_string(body.get_string_from_utf8())
    var action = int(json["action"]) # Extract the predicted values
    print("Action : ", action)
    
    # We put the code to move the robot here:
    parse_action_and_move(action)
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
func parse_action_and_move(action):
    await get_tree().process_frame
    # How much the bot moves per action passed.
    var bot_movement_forward_amount = 1
    var bot_rotation_amount = .5
    var fixed_forward_amount_special_action = 5
    
    if (action == 1):
        self.rotation.y += deg_to_rad(-bot_rotation_amount)
        robot_body.rotation.y += deg_to_rad(-bot_rotation_amount)
    elif (action == 0):
        self.rotation.y += deg_to_rad(bot_rotation_amount)
        robot_body.rotation.y += deg_to_rad(bot_rotation_amount)
    else:
        var fps = Engine.get_frames_per_second()
        var delta = 1.0 / fps
        
        if action == 3:
            # Special action to move forward a fixed amount
            for i in range(10):
                robot_body.global_transform.origin -= global_transform.basis.z * fixed_forward_amount_special_action * delta
                self.global_transform.origin -= global_transform.basis.z * fixed_forward_amount_special_action * delta
                await get_tree().process_frame
            return
        robot_body.global_transform.origin -= global_transform.basis.z * bot_movement_forward_amount * delta
        self.global_transform.origin -= global_transform.basis.z * bot_movement_forward_amount * delta
        await get_tree().process_frame

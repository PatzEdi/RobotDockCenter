from flask import Flask, request
from PIL import Image
import io
import base64
# Let's use sys.path to import the inference script from the machine_learning directory
import sys
import os

current_script_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(current_script_path,"../model"))
import inference  # Your inference script


class BotMovementCriteria:
    def __init__(self, x_distance_switch, y_distance_switch, x_distance_center):
        # For center box (We don't need a y distance for the center box, as
        # anyting within the x distance centered but not in the switch box
        # is considered in the center box and therefore a move forward action.
        # A triangle is not needed to compensate for distances, as if the
        # target is far away and in the center box, as we continue to move
        # forward it will go outside of the center box, making us turn toward
        # it.
        self.x_bound_center = (256 - x_distance_center, 256 + x_distance_center)
        # Note: the reason why we have separate x bounds for the two boxes is
        # because the center box will be slimmer, as the switch box has a
        # closer perspective and thus as allow for a wider range of x values.

        # For switch box
        # x_bound is the plus/minus from the center
        # of the image in the x direction (horizontal)
        self.x_bound_switch = (256 - x_distance_switch, 256 + x_distance_switch)
        # y_bound is the plus/minus from the center
        # of the image in the y direction (vertical).
        # This way, we create a "box"
        self.y_bound_switch = y_distance_switch

        # Other instance vars:
        self.inference = inference.InferenceTools()
        self.inference.load_model(1)  # Load the model
        self.changed_model = False
        self.target_num = 1 # used for get_is_target_detected

    def is_in_switch_area(self, x, y):
        """ Check if the target is in the switch area
        If it is, then we switch to the second model"""
        return self.x_bound_switch[0] < x < self.x_bound_switch[1] and y < self.y_bound_switch


    def is_in_center_box(self, x, y):
        """ Check if the target is in the center box
        If it is, then we move forward"""
        return self.x_bound_center[0] < x < self.x_bound_center[1]


    def get_rotation_code(self, x):
        """ Get the rotation code based on the x coordinate of the target"""
        if x < 256:
            return 0

        return 1


    def get_is_target_detected(self, image):
        # TO BE IMPLEMENTED SOON...(this function
        # simulates the yolo model by using opencv.
        # We can simulate the yolo mechanisim we will use
        # this way as we are in the virtual world, so even
        # opencv can detect the targets and check if they are there.)
       # Note: The image parameter is a PIL image.
       # NOTE we return true for now, as we have no implemented
       # this yet
       return True
       pass


    def get_bot_action(self, x, y):
        """ This function contains all of our logic for deciding what the
        next action should be. It should return either 0 (rotate right),
        1 (rotate left), or 2 (move forward)."""

        # NOTE: We will implement a yolo model to see if the targets
        # are actually there. For now though, we will simulate
        # the yolo model using opencv, and see if opencv detects the targets
        # (using the same method used in target_extraction.py). This will go
        # above this function call, as if the targets are not detected, we
        # rotate the bot until the targets are detected.
        # e.g. if len(target) == 0: return 0 # rotate right

        if not self.changed_model and self.is_in_switch_area(x, y):
            # NOTE: We also need to implement the
            # functionality to move forward a fixed amount
            self.inference.load_model(2)  # Load the model
            self.changed_model = True
            print("Switched to model 2!")
            self.target_num = 2 # target is now the dock
        elif self.is_in_center_box(x, y):
            return 2

        # If we are not in the center box or
        # in the switch box, we rotate
        return self.get_rotation_code(x)


class BotFlaskApp(BotMovementCriteria):
    def __init__(self, x_distance, y_distance, x_distance_center):
        super().__init__(x_distance, y_distance, x_distance_center)
        # Inference tools class
        self.app = Flask(__name__)
        self.set_app_routes()

    def set_app_routes(self):
        @self.app.route("/", methods=["GET"])
        def home():
            return {"message": "Server is running"}
        @self.app.route("/get_action", methods=["POST"])
        def get_action():
            """ Returns the next action that the virtual robot should take

            These actions will likely be used to control the robot in the
            real world as well. For now though this is just for the
            simulation

            Returns either: 0 (rotate right) 1 (rotate left)
            2 (move forward)"""
            # Get base64 image data from POST request
            base64_image_data = request.json["image"]
            # Decode base64 string to bytes
            image_data = base64.b64decode(base64_image_data)
            # Convert image data to PIL Image
            image = Image.open(io.BytesIO(image_data))
            # Run your inference script
            outputs = self.inference.predict(image)
            predicted_values = self.inference.parse_outputs(outputs)

            # Here we decide what model to use based on the position of the
            # coordinate for target 1. Think of it as a "box" at the bottom,
            # where if the coordinate is within those bounds, we switch to
            # model 2. Remember though that in doing so we also need to move
            # forward a fixed amount to ensure we are on top of target 1
            # instead of in front. I.e. the logic for deciding which model
            # to use goes here.
            # Here we implement the yolo simulation via opencv to see if the targets are
            # actually there. If they are not, we rotate the bot until they
            # are detected.
            if not self.get_is_target_detected(image):
                # If the current target is not detected, we rotate the bot
                # until it is detected
                action = 0
            else:
                action = self.get_bot_action(predicted_values[0], predicted_values[1])

            return {"action": action}

    # Run the app
    def run(self, debug=False):
        self.app.run()

if __name__ == "__main__":
   app = BotFlaskApp(40, 80, 20)
   app.run(debug=False)

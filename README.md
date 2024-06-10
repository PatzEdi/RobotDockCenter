# RobotDockCenter
 Applying machine learning techniques to create an automatic charge docking system for robots.

This project utilizes the Godot game engine in order to retrieve image data. It aims to automate the docking of robots into their charging stations.

**Note:** The machine learning phase for this project has started. However, the get_surpassed_images() method still has not yet been implemented, meaning images (and therefore instances) where the robot is either too far forward and not centered, or too far right or left when at the reverse line point, still are not taken into account for. For now, machine learning tests will be uses without that class of image just to see how the overall model performs.
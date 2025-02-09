# Table of Contents
1. [Introduction & General Methodology](#introduction--general-methodology)
2. [Review of the Old Method](#review-of-the-old-method)
3. [The New Method](#the-new-method)
4. [New vs Old Method: The Benefits & the Caveats](#new-vs-old-method-the-benefits--the-caveats)
5. [Using Multiple Models vs. One Yolo Model](#using-multiple-models-vs-one-yolo-model)
6. [Conclusion](#conclusion)

# Introduction & General Methodology
*Please read the note below before continuing.*

This document is an overview of the new method, as well as the old method used in this project. First, I would like to point out that each method has its own benefits and caveats, which I will discuss in the section [New vs Old Method: The Benefits & the Caveats](#new-vs-old-method-the-benefits--the-caveats).
However, each method has one distinct similarity: they both use a target system to guide the robot to the dock. This target system is a way to guide the robot to the dock, and is a very important part of the project.

**How does the target system work?**
1. The first target, also typically referenced throughout the project as target 1, is a point that lies a certain distance away from the center of the dock. This point is used to guide the robot to the dock. The second target, also typically referenced throughout the project as target 2, is a point that lies at the center of the dock. This point is used to guide the robot to the center of the dock.
2. Once we reach this first target, we then lock onto the second target, also typically referenced throughout the project as target 2, which represents the dock.

There are a few important questions to answer:
1. **How** do we decide which type of movement to make?
2. **When** do we decide to switch from the first target to the second target?

These are realities that in a system without reinforcement learning, we must answer with a set of rules, knowing that there will be some caveats to this system.

This system, however, is aimed at being light-weight, and is also aimed at using a simple input-output system in an efficient manner, making use of little complexity as possible.

Our main goal is to show how even the most basic of machine learning models can be used to automate a robot docking system with much more relaibility than standard image-processing algorithms through e.g. opencv, which also helps understand the vast possibilities of machine learning regarding greater complexity.

I myself aim to express all my ideas in the paper that I am writing, which I hope to release soon. You can check it out [here](/paper/tex/RobotDockCenter.pdf), it is directly in this repository. It is still in its very early stages though, and will take lots of time to finish and refine, so I hope you find value in my improvements over time.

## Review of the Old Method
The old method had two main values that were gathered as labels, or i.e. ground truth labels to train our model:

1. **The rotation angle of the robot**
2. **The distance of the robot from the center of the dock (represented by a virtual 'line').**

These two values are very simple to get in a game engine. In the real world, however, we would need to measure the distance of the robot from the center of the dock (represented by a 'virtual' line in the middle of the dock), and the angle the robot would need to turn (in degrees).

This would take months of data labeling, especially considering the fact that we had used around 6000 images to train our model.

Diving a bit deeper into how the old method worked (while keeping this explanation brief), we would set a threshold to the rotation value. If the rotation value was within a certain range, we would move the robot forward. Else, we would rotate according to the sign of the rotation (in degrees).
Considering the second value about the distance from the center, we would move the robot forward when the rotation value was within the threshold (for each frame), and when the distance from the center reached a certain amount, we would stop the robot and switch targets from the first target, to the second target.

**This idea of two targets** is shared with the new method, but the way we get the values is different.
Besides the unpracticality of the old method, it had a few advantages, of which we compare further when talking about [New vs Old Method: The Benefits & the Caveats](#new-vs-old-method-the-benefits--the-caveats).


# The New Method
Previously, as reviewed above, the robot docking that we used essentialy relied solely on game engine data that I realized would be very impractical to obtain in the real world. This new method described below is a step in the right direction, as it is more realistic to obtain data for it in the real world, while maintaining docking reliability.

Instead of having a rotation value and a distance from the center value, we extract the coordinates of the specified target from the image. The coordinates make use of the image as a 2D plane.

In doing so, we have solved the first question about which movement we should make. Using the coordinates, we can calculate the direction the robot must turn, and the threshold for the coordinates to condider the movement as having to move forward.

Using this coordinate system, we can also answer the second question about when to switch targets. You see, if you have coordinates in the image, with the image acting as a 2D plane, we can set a threshold on the bottom of the image. Think of it as a "box".

Once the coordinates are within this "box" it is apparent that target 1 has been reached, and we can proceed to target 2. There is one thing to keep in mind, however:
**When the predicted target coordinates enter that "box" to switch targets, we must also move the robot forward a fixed amount**

This is because of the fact that we want the robot to be on top of target 1 in order for it to be centered with the dock. Otherwise, we would still be slightly off.

So in short, instead of two different values, we use the coordinates of the target in the image to make the robot's movements. This makes everything simpler, but for a deeper comparison of the two methods and their caveats, please read the final section below: [New vs Old Method: The Benefits & the Caveats](#new-vs-old-method-the-benefits--the-caveats).

# New vs Old Method: The Benefits & the Caveats

## The New Method: Benefits & Caveats
The new method has many benefits:
1. Easily translatable to the real world. The coordinates system is easy to translate in the real world, as all that would need to be done is label the point of where our target is.
2. Simplicity. The new method simpler than the old method in terms of handling ground truth labels. This makes training much easier.
3. The target system can be implemented using a single yolo model! Yes, that is right, this entire method would work using a single yolo model, although I have not tried that yet. In theory, though, it would work. More about that in [Using Multiple Models vs. One Yolo Model](#using-multiple-models-vs-one-yolo-model).
4. The new method maintains reliability of the old method, while being more realistic to obtain data for in the real world.

Relating to the translating of the functionality of the new method to the real world, the same goes for the training process. What we did to train the model using synthetic data can be done in the real world, but instead of using opencv to detect the targets, we would manually label them (as real images have many more colors that aren't present in the synthetic world). The reason why we can use opencv to get the "ground truth" labels is because we detect green (target 1) and red (target 2), in the synthetic world, that has no other objects with these colors. So, the only thing that would change would be the way we label the data, which would be easily done by hand. The training process, however, remains the same, with of course, the images that change.

The new method, however, has a few caveats:
1. We do not know whether or not we are centered with the dock until we reach the first target. This is because we rely on the first target to be reached first, and then we move forward a fixed amount to be centered with the dock. This is the biggest caveat of the new method. Like I said, however, it is a trade-off for the simplicity of the new method while not using reinforcement learning.
2. The new method requires a more apparent indicator for target 1. This is because we, as humans, need a clear reference point when labeling the ground truth data. Without a clear reference to target 1, we cannot expect reliable results from the model.

## The Old Method: Benefits & Caveats
First, let's discuss the benefits of the old method.
The old method had many benefits, of which the new method does not have. These benefits are:
1. Using the distance from the center of the dock, we could determine whether or not we were centered with the dock. With the new method, as desscribed above, this is not possible, as we rely on the first target to be reached first.
2. The old method did not require a more apparent inidicator for target 1. While this may seem like a "benefit", it is actually unrealistic in the real world. In order to label data effectively, humans must have a consistent reference point as to what they want the model to learn. If this method worked, it would remove the extra work of the first target having to be more apprent in the docking environment.

The biggest advantage over the new method is the fact that we would be able to determine whether or not we are centered with the dock at any point during the procedure. The new method, however, expects to be centered once reaching the first target, which means that even if it is directly in front of the dock and centered, it will not know it is centered.

The real major caveat with the old method was that it required data labeling that in the real world is simply not plausible to obtain. For example, in order to get the distance from the robot to the center of the dock, we would need to measure this image by image. This is the same for the rotation value. We would need to get the rotation in degrees for each image required to turn towards the dock.

I want to emphasize the importance of reinforcement learning in robots nowadays. This project is an example of how you don't need to use reifnorcement learning for a set of simple rules regarding docking. In more advanced systems, however, simulations are necessary in order to gather synthetic data that can be used to translate the model's knowledge to the real world.

# Using Multiple Models vs. One Yolo Model
The current method uses two models. In practice, however, it would use three: a Yolo model. Let me explain this unconventional way of completing the dock process, and then intoruce a more plausible and simple approach:
The first model predicts the coordinates of target 1. The second model predicts the coordinates of target 2. The third model I will explain after I have explained the first two:
The reason why I first am using this two-model system is because I wanted to create custom models. In doing so, I could experiment better with model complexity, and determine what the most conservative approach to the problem was. I tried to complete the problem with the least amount of model complexity, and sure enough, I was about to complete it with very small models.
Anyways, the first model outputs the coordinates of target 1. One we have reached the "switch" point, we then load the second model to target the dock. This is a very unconventional way of completing the dock process, but it works. The third model, however, would be a single yolo model that would predict the coordinates of both target 1 and target 2.
The third model in such a system would be a yolo model. It would scan for the models every certain n amount of frames, to make sure that our target is still there. This would pair well for e.g. the start of the process, where we want to rotate the robot to search for the dock.

**Now, let's talk about a more plausible and simple approach:**
The more plausible and simple approach uses a single Yolo model. We could define which target we want to go to, and the yolo model would predict the bounding box. We could then take the bounding box and calculate the average of its coordinates.

Using a single yolo model, we combine the two models into one, with the feature of the third model, which would be its ability to scan whether or not the target is in the frame.

*I have not* started this implementation yet, and doing so would require fine tuning a pretrained yolo model. I am excited to see how this would work, and I am sure that it would work well.

# Conclusion
The new method is a step in the right direction, as it is more realistic to obtain data for in the real world, while maintaining docking reliability. This new method allows us to better explore the problem in the real world by making the process realistic. Also, the new method translates directly from the synthetic world to the real world. By synthetic world, I mean the virtual docking environment shown in the README video.
I am excited to see how this project will unfold, and am currently writing the complete explanation for it in the paper. I hope you find value in my improvements over time, and I am excited to share my thoughts with you.
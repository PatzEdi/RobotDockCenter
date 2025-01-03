# RobotDockCenter
 Applying machine learning techniques to create an automatic charge docking system for robots.

This project utilizes the Godot game engine in order to retrieve image data. It aims to automate the docking of robots into their charging stations.

## **Current Progress:**
A new method is being implemented. It is not described here yet as I have planned it out on paper. So, future commits will likely be centered around this new method.
If I see that it is feasible, I will include the method here or in a separate document.

## **Current Godot Engine Version Used:**
**4.3**

**Docking Demo (7-14-24)**

[DockingDemo](https://github.com/user-attachments/assets/17d77959-a484-4489-93ab-d440e66e084c)

**Current Data Info:**

```
6630 images (for each model)
```

**Loss Curve (healthy):**
✅

![Alt text](assets/latest_loss_curve.jpg "Current Loss Curve")

**trained with:**
```
learning_rate = 0.001 (or .0005)
batch_size = 16
num_epochs = 15

weight_distance = .85
weight_rotation_value = .9
```

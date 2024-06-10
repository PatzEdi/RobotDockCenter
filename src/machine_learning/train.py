# Let's import the necessay libraries for image processing and model training. Use pytorch:
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
# We import the functions and variables from data_process.py used here for training:
from data_process import get_data_targets
from data_process import image_paths

data_targets = get_data_targets(torch,True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# The custom dataset class used later in the Dataloader:
class TargetsDataset(Dataset):
    def __init__(self, image_paths, data_targets, transform=None):
        self.image_paths = image_paths
        self.data_targets = data_targets # The full list of lists that contains the target values for each image
        self.transform = transform
    def __len__(self):
        return len(self.image_paths) # Return the amount of images in the data set, thus the amount of samples to train/iterate over
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])  # Use PIL to open the image

        direction = self.data_targets[idx][0].unsqueeze(0) # Get the direction tensor data sample
        distance_rline = self.data_targets[idx][1].unsqueeze(0) # Get the distance_rline tensor data sample
        rotation_value = self.data_targets[idx][2].unsqueeze(0) # Get the rotation_value tensor data sample

        if self.transform:
            image = self.transform(image)

        return image, [direction, distance_rline, rotation_value] # Return the image, as well as the target values in a list

# Define a simple neural network and simple dataset to test the coordinates prediction:
class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # The initial parameter 3 is used to define the number of channels (in this case three, because of (R,G,B))
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.direction = nn.Linear(32*32*64, 1) # Predict the direction value
        self.distance_rline = nn.Linear(32*32*64, 1) # Predict the distance value
        self.rotation_value = nn.Linear(32*32*64, 1) # Predict the rotation value

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        direction = self.direction(x)
        distance_rline = self.distance_rline(x)
        rotation_value = self.rotation_value(x)

        return [direction, distance_rline, rotation_value]

# Here is the training part:
if __name__=="__main__": # This is used to prevent the code below from running when calling this script from another script, specifically the inference.py script.
    """Let's create some hyperparameters and instantiate the dataset, dataloader, model, criterion, and optimizer, and then finally, the training loop"""
    learning_rate = 0.001
    batch_size = 16 # Leave at one for stochastic gradient descent
    num_epochs = 5
    
    # [Experimental] determine the importance of each loss value:
    weight_direction = .9
    weight_distance = .85
    weight_rotation_value = 1.0

    print(f"Learning Rate: {learning_rate}\nBatch Size: {batch_size}\nNum Training Epochs: {num_epochs}")
    print("\nUsing Stochastic Gradient Descent (Batch Size = 1)\n") if batch_size == 1 else print("\nUsing Mini-Batch Gradient Descent\n") # Print out the message that indicates whether or not we are using stochastic gradient descent.

    # Let's instantiate the dataset class, as well as the dataloader provided by PyTorch:
    targets_dataset = TargetsDataset(image_paths, data_targets, transform=transform)
    dataloader = DataLoader(targets_dataset, batch_size=batch_size, shuffle=True) # Batch size can't be greater than 1 because the coordinate lists are of varying sizes. Consider the padding method, and then using RNNs, LSTMs, or GRUs to then ignore the padding.

    print("Starting training stage...\n")
    # Let's instantiate the model, criterion, and optimizer:
    model = Predictor()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # We use Adam for adaptive learning rates rather the SGD (To be tested and experimented with in the future...)
    
    loss_values = [] # This list will be used to store the loss values for each epoch.
    # Here is the training loop:
    for epoch in range(num_epochs):
        for images, targets in dataloader:  # targets is a list: [coords, num_coords, distance]
            optimizer.zero_grad()

            outputs = model(images) # Retreive the model's output
            # print(outputs[0], targets[0])
            # exit()
            loss = weight_direction * criterion(outputs[0], targets[0]) + weight_distance * criterion(outputs[1], targets[1]) + weight_rotation_value * criterion(outputs[2], targets[2])
            loss.backward()
            optimizer.step()
        # Print some info:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss value: {loss.item()}")
        loss_values.append(loss.item())

    current_script_path = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_script_path, '../../models/predictor_model.pth')
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)

    print("\nTraining finished!")
    # Let's graph the loss values with plt:
    plt.plot(loss_values)
    plt.title("Loss Values Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.show()
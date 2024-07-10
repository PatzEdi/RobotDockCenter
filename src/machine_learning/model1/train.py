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
import sys
# For the progress bar:
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add the parent directory to the path so that we can import the data_process.py file
# We import the functions and variables from data_process.py used here for training:
from data_process import get_data_targets_model1 as get_data_targets
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

        rotation_value = self.data_targets[idx][0].unsqueeze(0) # Get the rotation_value tensor data sample
        distance_cline = self.data_targets[idx][1].unsqueeze(0) # Get the distance_cline tensor data sample

        if self.transform:
            image = self.transform(image)

        return image, [rotation_value, distance_cline] # Return the image, as well as the target values in a list

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
        self.fc1 = nn.Linear(32*32*64, 128)
        self.combined_output = nn.Linear(128, 2) # Output size is 2


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x)) # We use the ReLU activation function for the fully connected layer to introduce nonlinearity, with 128 neurons

        outputs = self.combined_output(x)
        rotation_value, distance_cline  = outputs.split(1, dim=1)

        return [rotation_value, distance_cline]

# Here is the training part:
if __name__=="__main__": # This is used to prevent the code below from running when calling this script from another script, specifically the inference.py script.
    """Let's create some hyperparameters and instantiate the dataset, dataloader, model, criterion, and optimizer, and then finally, the training loop"""
    learning_rate = 0.001
    batch_size = 16 # Leave at one for stochastic gradient descent
    num_epochs = 20
    

    weight_distance = .85
    weight_rotation_value = .9

    print(f"Learning Rate: {learning_rate}\nBatch Size: {batch_size}\nNum Training Epochs: {num_epochs}")
    print("\nUsing Stochastic Gradient Descent (Batch Size = 1)\n") if batch_size == 1 else print("\nUsing Mini-Batch Gradient Descent\n") # Print out the message that indicates whether or not we are using stochastic gradient descent.

    # Let's instantiate the dataset class, as well as the dataloader provided by PyTorch:
    targets_dataset = TargetsDataset(image_paths, data_targets, transform=transform)
    dataloader = DataLoader(targets_dataset, batch_size=batch_size, shuffle=True)
    print("Starting training stage...\n")
    # Let's instantiate the model, criterion, and optimizer:
    model = Predictor()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # We use Adam for adaptive learning rates rather the SGD (To be tested and experimented with in the future...)
    
    model.train()

    loss_values = [] # This list will be used to store the loss values for each epoch.
    # Here is the training loop:
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=85)
        for images, targets in dataloader:  # targets is a list: [rotation_value, distance_cline]
            optimizer.zero_grad()
            outputs = model(images) # Retreive the model's output
            loss = weight_rotation_value * criterion(outputs[0].double(), targets[0]) + weight_distance * criterion(outputs[1].double(), targets[1])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.update()

        progress_bar.close()
        
        average_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss value: {average_loss}\n")
        loss_values.append(average_loss)

    current_script_path = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_script_path, '../../../models/model1.pth')
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)

    print("\nTraining finished!")
    # Let's graph the loss values with plt:
    plt.plot(loss_values)
    plt.title("Loss Values Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.show()
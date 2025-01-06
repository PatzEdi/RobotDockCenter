import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
# For the progress bar:
from tqdm import tqdm

# ====================
# NOTE: This file is currently undergoing heavy refactoring.
# Also, this  is largely for the synthetic world, alhtough for the real world
# the implementation will be similar.
# ====================
current_script_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_script_path, "../../godot/data_images")

# Custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import config_parser
from target_extraction import get_images_for_each_target


# Resize each image to 128x128 pixels.
# This is mostly just to improve performance rather than
# ensure equal image sizes, as each image is already 512x512.
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# The custom dataset class used later in the Dataloader:
class TargetsDataset(Dataset):
    def __init__(self, target_image_data, transform=None):
        self.target_image_data = target_image_data
        self.transform = transform

    def __len__(self):
        # Return amount of images in the data set
        return len(self.target_image_data)

    def __getitem__(self, idx):
        image = Image.open(self.target_image_data[idx][0])

        # idx is the image index
        # 1 is the index of the tuple of targets
        # 0 or 1 for the final indexing is for the target value, either
        # x_avg or y_avg (the coords of the target point)
        x_coord = self.target_image_data[idx][1][0]
        y_coord = self.target_image_data[idx][1][1]

        targets = torch.tensor([x_coord, y_coord], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, targets # Return image w/targets

# Define a simple neural network
class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        self.features = nn.Sequential(
            # The initial parameter 3 is used to define the number of channels
            # (in this case three, because of (R,G,B))
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(32*32*64, 128)
        # Out output will be of size [N, 1, 2], where N is the batch size,
        # and each batch has a list of two values that correspond to the
        # x_avd and y_avg values of ground truth labels.
        self.fc2 = nn.Linear(128, 2)


    def forward(self, x):
        x = self.features(x) # get output of the convolutional layers
        # Flatten the output of the convolutional
        # layers
        x = x.view(x.size(0), -1)

        # We use the ReLU activation function for the fully
        #connected layer to introduce nonlinearity, with 128 neurons
        x = F.relu(self.fc1(x)) # Relu to hidden layer


        return self.fc2(x) # Return the output of the fully connected layer

class Train:
    # Let's add a train function here to train the model for organization
    @staticmethod
    def train(target_image_data, n_epochs, lr, batch_size):

        # Print some general information about the training process
        print(
            f"Learning Rate: {lr}\n"
            f"Batch Size: {batch_size}\n"
            f"Num Training Epochs: {n_epochs}"
        )

        # Let's instantiate the dataset class and the Pytorch Dataloader:
        targets_dataset = TargetsDataset(
            target_image_data,
            transform=transform
        )

        dataloader = DataLoader(
            targets_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        print("Starting training stage...\n")
        # Let's instantiate the model, criterion, and optimizer:
        model = Predictor()
        criterion = nn.MSELoss()
        # We use Adam for adaptive learning rates rather the SGD (To be tested
        # and experimented with in the future...)

        # Use Adam for adaptive learning rates to avoid vanishing gradients
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        # This list will be used to store the loss values for each epoch.
        loss_values = []
        # Here is the training loop:
        for epoch in range(num_epochs):
            total_loss = 0.0
            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                ncols=85
            )
            # targets is a list: [x_coord, y_coord]
            for image, targets in dataloader:
                # Zero the gradients (to avoid accumulation, as Pytorch)
                optimizer.zero_grad()
                outputs = model(image) # Retreive the model's output
                loss = criterion(outputs, targets) # Here we calculate the loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                progress_bar.update()

            progress_bar.close()

            average_loss = total_loss / len(dataloader)
            print(
                f"Epoch {epoch+1}/{num_epochs},"
                f"Average Loss value: {average_loss}\n"
            )
            # We append to the list for later plotting
            loss_values.append(average_loss)

        model_save_path = os.path.join(
            current_script_path,
            "../../../models/model" + str(model_num) + ".pth"
        )
        # Save the trained model
        torch.save(model.state_dict(), model_save_path)

        print("\nTraining finished & model has been saved!")

        return loss_values


# Here is the training part:
# This is used to prevent the code below from running when calling this script
# from another script, specifically the inference.py script.
if __name__=="__main__":

    # We get the data
    print("Getting & preprocessing data...")
    # Target 1 is for the green target, and target 2 is for the red target
    # TODO: We need to change this so that only one target data is loaded,
    # so that time and memory is saved.
    target_1_data, target_2_data = get_images_for_each_target(data_dir)

    model_num = int(config_parser.get_model_num())

    print("\nTraining Model Number " + str(model_num))
    # Set the data var based on the model we are training
    target_image_data = target_1_data
    if model_num != 1:
        target_image_data = target_2_data

    """Let's create some hyperparameters and instantiate components.

    Components include the dataset, dataloader, model, criterion, and optimizer
    , and then finally, the training loop
    """
    learning_rate = 0.0005
    batch_size = 8 # Leave at one for stochastic gradient descent
    num_epochs = 15

    # We train the model and get the loss values in doing so
    loss_values = Train.train(
        target_image_data,
        num_epochs,
        learning_rate,
        batch_size,
    )

    # Let's graph the loss values with plt:
    plt.plot(loss_values)
    plt.title("Loss Values Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.show()

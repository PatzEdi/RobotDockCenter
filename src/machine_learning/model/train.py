import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
# For the progress bar:
from tqdm import tqdm


def get_device():
    # Let's define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check that MPS is available (for Macs with M chips)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    # We return the device
    return device

# ====================
# This  is largely for the synthetic world, alhtough for the real world
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
    # Let's random grayscale the image with a probability of 0.5
    transforms.RandomGrayscale(p=0.5),
    transforms.ToTensor()
])

# Let's create a function that performs data augmentation
# by flipping the image vertically randomly with a probability.
# We need to make this custom as when we do this, the coordinates
# need to be flipped as well.
def custom_vertical_flip(p, image, targets):
    """ Flip the image vertically with a probability p.
    Targets is expected to be in the form a tensor of size [1, 2],
    with normalized (divided by 512) target coords."""
    if torch.rand(1) < p:
        # We flip the image vertically
        image = F2.vflip(image)
        # We need to flip the y coordinate, but not the x coordinate
        # as the x coordinate is the same.
        targets[1] = abs(1 - targets[1])
    return image, targets


def custom_horizontal_flip(p, image, targets):
    """ Flip the image horizontally with a probability p.
    Targets is expected to be in the form a tensor of size [1, 2],
    with normalized (divided by 512) target coords."""
    if torch.rand(1) < p:
        # We flip the image horizontally
        image = F2.hflip(image)
        # We need to flip the x coordinate, but not the y coordinate
        # as the y coordinate is the same.
        targets[0] = abs(1 - targets[0])
    return image, targets


# The custom dataset class used later in the Dataloader:
class TargetsDataset(Dataset):
    def __init__(self, target_image_data, transform=None, augment=True):
        self.target_image_data = target_image_data
        self.transform = transform
        self.augment = augment

    def __len__(self):
        # Return amount of images in the data set
        return len(self.target_image_data)

    def __getitem__(self, idx):
        image = Image.open(self.target_image_data[idx][0])

        # idx is the image index
        # 1 is the index of the tuple of targets.
        # It is already a tensor
        targets = self.target_image_data[idx][1]

        if self.transform:
            image = self.transform(image)

        if self.augment:
            # Data augmentation
            image, targets = custom_vertical_flip(.5, image, targets)

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
    def train(target_image_data, n_epochs, lr, batch_size, model_num, augment):

        print(f"\nTotal Training Images: {len(target_image_data)}")
        # We get the device
        device = get_device()
        # Print some general information about the training process
        print(
            f"Learning Rate: {lr}\n"
            f"Batch Size: {batch_size}\n"
            f"Num Training Epochs: {n_epochs}"
        )

        # Let's instantiate the dataset class and the Pytorch Dataloader:
        targets_dataset = TargetsDataset(
            target_image_data,
            transform=transform,
            augment=augment
        )

        dataloader = DataLoader(
            targets_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        print("Starting training stage...\n")
        # Let's instantiate the model, criterion, and optimizer:
        model = Predictor()
        model.to(device)
        criterion = nn.L1Loss()
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
                image = image.to(device)
                targets = targets.to(device)
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
        # Move to cpu to make it easier to load later
        model.cpu()
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
    # so that time and memory is saved. Also, for now, we load all the images
    # available. In the future, we will split the data into training and
    # testing sets.
    target_1_data, target_2_data = get_images_for_each_target(
        data_dir,
        n_images_cls=-1
    )

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
    batch_size = 32 # Leave at one for stochastic gradient descent
    num_epochs = 20

    # We train the model and get the loss values in doing so
    loss_values = Train.train(
        target_image_data,
        num_epochs,
        learning_rate,
        batch_size,
        model_num,
        augment=True
    )

    # Let's graph the loss values with plt:
    plt.plot(loss_values)
    plt.title("Loss Values Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.show()

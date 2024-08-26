import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import logging
import time
import matplotlib.pyplot as plt
import numpy as np


from extendedCNN import SlidingProjectionNet

from classes import *

# Setup logging
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"./log/{timestamp}_training_log.log"
logging.basicConfig(filename=log_filename, level=logging.INFO)


# In case of continuos training or fine tuning
model_to_load = None #"checkpoint_20240807_110214.pth"



# TODO This can be moved to argument
dataset = "FashionMNIST" # CIFAR10 | FashionMNIST
dataset_path = '../data' # Dataset will be saved or used from here - I have it in upper level due to other models

# model parametrs
if dataset =="FashionMNIST":
    start_channels = 1
else:
    start_channels = 3 #For CIFAR10 at this moment

num_classes = 10    # Fix for now for both CIFAR10 and FashionMNIST

# global data
learning_rate = 0.001

# Define the normalization transform based on the number of channels
if start_channels == 1:
    normalize_transform = transforms.Normalize((0.5,), (0.5,))  # For grayscale images
else:
    normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For RGB images


# Data augmentation transforms
train_transform = transforms.Compose([
    # TODO move to args what kind of augmentioation we want to do
    #transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    #transforms.RandomCrop(32, padding=4),  # Randomly crop images
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
    
    transforms.ToTensor(),  # Convert images to tensor
    
    # Normalize the images based on the chanells
    normalize_transform
])

# No data augmentation for validation/testing, only normalization
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_transform
])


if dataset =="FashionMNIST":
    trainset = torchvision.datasets.FashionMNIST(root=dataset_path, train=True, download=True, transform=train_transform)
    validset = torchvision.datasets.FashionMNIST(root=dataset_path, train=False, download=True, transform=test_transform)
else:
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
    validset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=False)

# Just log the data
logging.info(dataset) 

# Instantiate the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SlidingProjectionNet(start_channels=start_channels, num_classes=num_classes).to(device)
# Load model and optimizer state in case of continuing and/or fine tuning
if model_to_load:
    checkpoint = torch.load(f'./model/{model_to_load}')
    model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss()
# Optimizer should be checked and modified. SGD or AdamW are returning better values sometimes
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)
# In case we want to continue in training on more epochs, load the optimizer
if model_to_load:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Training the model
num_epochs = 5 # TODO move to arguments
train_losses = []
valid_losses = []
valid_accuracies = []

for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Start time of the epoch
    # Not using any dynamic LR at this train file
    # TODO can be made better to use some dynamic LR algos.
    # TODO can be for warmup, oscilating, etc. For this we need to enable early stopping
    if epoch > 0 and epoch <4:
            for param_group in optimizer.param_groups:
                #param_group['lr'] *= 2 # Increase the lr from 0.001 -> 0.01
                pass
        
    elif epoch == 5:
        for param_group in optimizer.param_groups:
            #param_group['lr'] = learning_rate # Return back to the original lr
            pass
            """ if optim_algo == "SGD":
                param_group['momentum'] = 0.95 #  Increase momentum """
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_losses.append(running_loss / len(trainloader))
    
    # Validation step - at this moment validation after each epoch. 
    # TODO - !!! This is would not be great for a larger epochs
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_dot_products = []
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, feature_maps = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate dot products
            """ dot_products = calculate_dot_product(feature_maps)
            all_dot_products.append(dot_products) """
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    valid_losses.append(val_loss / len(validloader))
    val_accuracy = 100 * correct / total
    valid_accuracies.append(val_accuracy)

    # Calculate Precision, Recall, F1-Score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Save or visualize the dot products
    """ for idx, dot_product_set in enumerate(all_dot_products):
        for layer_idx, dot_product in enumerate(dot_product_set):
            torch.save(dot_product.cpu(), f"./dp/{timestamp}_dot_product_layer{layer_idx+1}_batch{idx+1}.pt") """

    # Calculate the time taken for the epoch
    epoch_duration = time.time() - epoch_start_time  # Calculate the duration of the epoch
    
    
    log_message = (f"Epoch {epoch+1}, Train Loss: {running_loss/len(trainloader):.4f}, "
               f"Validation Loss: {val_loss/len(validloader):.4f}, Validation Accuracy: {val_accuracy:.2f}%. "
               f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}. "
               f"Time: {epoch_duration:.2f} seconds, "
               f"LR: {optimizer.param_groups[0]['lr']},")
    
    
    

    print(log_message)
    logging.info(log_message)
    
    # Step the scheduler
    scheduler.step(val_loss / len(validloader))

    # Early Stopping Check # TODO not used in this now
    """ if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        trigger_times += 1
        if trigger_times >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs.")
            break """

checkpoint_name = f"./model/checkpoint_{timestamp}.pth"

# Save model and optimizer state into a checkpoint TODO not saving now
# TODO saving can be mofed to args
""" torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, checkpoint_name) """





# ANALYZE section




# Load the input image and label
input_image, label = validset[0]  # Get the first sample and its label

# Move input image to the device
input_image = input_image.to(device)

# Get the feature maps and output
feature_maps = visualize_feature_maps(model, input_image)


# Convert the input image for display
input_image_np = input_image.cpu().numpy()  # Move to CPU
# Check if the image is grayscale or RGB
if input_image_np.shape[0] == 1:
    # Grayscale image (FashionMNIST)
    input_image_np = input_image_np.squeeze(0)  # Remove channel dimension
else:
    # RGB image (CIFAR-10)
    input_image_np = input_image_np.transpose(1, 2, 0)  # Rearrange dimensions to (H, W, C)

input_image_np = (input_image_np * 0.5) + 0.5  # Denormalize assuming normalization with mean=0.5, std=0.5
input_image_np = np.clip(input_image_np, 0, 1)  # Clip to ensure values are within [0, 1]

# Display the input image
plt.figure(figsize=(5, 5))
plt.imshow(input_image_np, cmap='gray')  # Change to 'viridis' if the input image is RGB
plt.title(f"Input Image - Label: {label}")
plt.axis('off')
plt.savefig(f"./fm/{timestamp}_input_{label}.png")
#plt.show()

# Visualize the feature maps
for idx, fmap in enumerate(feature_maps):
    # Plot each feature map for a layer
    fmap = fmap.squeeze(0)  # Remove batch dimension
    fmap = fmap.cpu().numpy()  # Move to CPU and convert to numpy array
    
    num_feature_maps = fmap.shape[0]
    
    plt.figure(figsize=(15, 15))
    for i in range(num_feature_maps):
        plt.subplot(8, 8, i + 1)  # Adjust the grid size as needed
        plt.imshow(fmap[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f"Layer {idx + 1} Feature Maps")
    plt.savefig(f"./fm/{timestamp}_fm_layer_{idx + 1}.png")
    #plt.show()




# Define the class names
if dataset == "FashionMNIST":
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']     # FashionMNIST
else:
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']                     # CIFAR10


# Save the confusion matrix with real labels
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig(f"./log/{timestamp}_confusion_matrix.png")
#plt.show()

# Error analysis: Display some wrongly classified images with real labels
wrong_preds = [(img, pred, true) for img, pred, true in zip(validset.data, all_preds, all_labels) if pred != true]
plt.figure(figsize=(15,15))
for i, (img, pred, true) in enumerate(wrong_preds[:25]):
    plt.subplot(5, 5, i + 1)
    plt.imshow(img)
    plt.title(f"True: {class_names[true]}\nPred: {class_names[pred]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"./log/{timestamp}_error_analysis.png")
#plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from torch.utils.data import DataLoader, Subset
from center_loss import CenterLoss
from models.resnet34 import Resnet34

batch_size = 15
num_workers = 2
validation_dataset_split_ratio = 0.1

embedding_dimension = 128
pretrained = False

learning_rate = 0.001
learning_rate_center_loss = 0.5
alpha = 10
center_loss_weight = 0.5

# Define transforms for the training and test sets
data_transforms = transforms.Compose([
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the dataset with ImageFolder
dataset = torchvision.datasets.ImageFolder(
    root="/home/tamer/University Courses/__Graduation Project/Datasets/original_datasets/CASIA-WebFace/",
    transform=data_transforms
)

# Subset the dataset into training and validation datasets
num_classes = len(dataset.classes)
print("Number of classes in dataset: {}".format(num_classes))
num_validation = int(num_classes * validation_dataset_split_ratio)
num_train = num_classes - num_validation
indices = list(range(num_classes))
np.random.seed(42)
np.random.shuffle(indices)
train_indices = indices[:num_train]
validation_indices = indices[num_train:]

train_dataset = Subset(dataset=dataset, indices=train_indices)
validation_dataset = Subset(dataset=dataset, indices=validation_indices)
print("Number of classes in training dataset: {}".format(len(train_dataset)))
print("Number of classes in validation dataset: {}".format(len(validation_dataset)))


# Define the dataloaders
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers
)

validation_dataloader = DataLoader(
    dataset=validation_dataset,
    batch_size=batch_size,
    num_workers=num_workers
)


model = Resnet34(
    num_classes=num_classes,
    embedding_dimension=embedding_dimension,
    pretrained=pretrained
)


flag_train_gpu = torch.cuda.is_available()
flag_train_multi_gpu = False

if flag_train_gpu and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.cuda()
    flag_train_multi_gpu = True
    print('Using multi-gpu training.')
elif flag_train_gpu and torch.cuda.device_count() == 1:
    model.cuda()
    print('Using single-gpu training.')


criterion_crossentropy = nn.CrossEntropyLoss().cuda()
criterion_centerloss = CenterLoss(num_classes=num_classes, feat_dim=embedding_dimension, use_gpu=True)

# Optimize model parameters and the center loss object parameters
parameters = list(model.parameters()) + list(criterion_centerloss.parameters())
optimizer = optim.Adam(parameters, lr=learning_rate)


learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer,
    milestones=[40, 70],
    gamma=0.1
)

# Training loop
epochs = 100

print("\nTraining starting for {} epochs:\n".format(epochs))
total_time_start = time.time()

for epoch in range(epochs):
    train_loss = 0
    validation_loss = 0

    epoch_time_start = time.time()
    # Training the model
    learning_rate_scheduler.step()
    model.train()
    for data, labels in train_dataloader:
        data, labels = data.cuda(), labels.cuda()
        # Forward pass
        if flag_train_multi_gpu:
            embedding, logits = model.module.forward_training(data)
        else:
            embedding, logits = model.forward_training(data)
        # Calculate losses
        cross_entropy_loss = criterion_crossentropy(logits.cuda(), labels.cuda())
        center_loss = criterion_centerloss(embedding, labels)
        loss = (center_loss * center_loss_weight) + cross_entropy_loss
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        for parameter in criterion_centerloss.parameters():
            # Remove the effect of alpha (center_loss_weight) on updating centers
            parameter.grad.data *= (learning_rate_center_loss / (alpha * learning_rate))
        optimizer.step()
        # Update average training loss
        train_loss += loss.item()*data.size(0)

    # Testing the model
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in validation_dataloader:
            data, labels = data.cuda(), labels.cuda()
            # Forward pass
            if flag_train_multi_gpu:
                embedding, logits = model.module.forward_training(data)
            else:
                embedding, logits = model.forward_training(data)
            # Calculate losses
            cross_entropy_loss = criterion_crossentropy(logits.cuda(), labels.cuda())
            center_loss = criterion_centerloss(embedding, labels)
            loss = (center_loss * center_loss_weight) + cross_entropy_loss
            # Update average validation loss
            validation_loss += loss.item() * data.size(0)
            # Calculate training performance metrics
            predictions = logits.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    # Calculate average losses in epoch
    train_loss = train_loss / len(train_dataloader.dataset)
    validation_loss = validation_loss / len(validation_dataloader.dataset)

    # Calculate training performance statistics in epoch
    accuracy = correct * 100. / total
    error = 100. - accuracy

    epoch_time_end = time.time()
    # Print training and validation statistics
    print('Epoch {}:\tTraining Loss: {:.4f}\tValidation Loss: {:.4f}\tValidation Accuracy: {:.2f}%\t\
        Validation Error: {:.2f}%\tEpoch Time: {:.2f} minutes'.format(
        epoch+1, train_loss, validation_loss, accuracy, error, (epoch_time_end - epoch_time_start)/60
    ))

    # Save model checkpoint
    if flag_train_multi_gpu:
        state = {
            'epoch': epoch,
            'num_classes': num_classes,
            'batch_size_training': batch_size,
            'model_state_dict': model.module.state_dict(),
            'model_architecture': 'resnet34',
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'validation_accuracy': accuracy,
            'validation_error': error,
            'elapsed_training_time_seconds': time.time() - total_time_start
        }
    else:
        state = {
            'epoch': epoch,
            'num_classes': num_classes,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': 'resnet34',
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'validation_accuracy': accuracy,
            'validation_error': error,
            'elapsed_training_time_seconds': time.time() - total_time_start
        }

    torch.save(state, 'model_resnet34_vggface2.pt')

# Training loop end
total_time_end = time.time()
total_time_elapsed = total_time_end - total_time_start

print("\nTraining finished: total time elapsed: {:.2f} minutes.".format(total_time_elapsed/60))

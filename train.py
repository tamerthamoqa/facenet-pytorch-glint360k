import numpy as np
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from torch.utils.data import DataLoader, Subset
from center_loss import CenterLoss
from LFWDataset import LFWDataset
from validate_on_LFW import evaluate_lfw, plot_roc_lfw
from tqdm import tqdm
from models.resnet34 import Resnet34
from models.resnet50 import Resnet50
from models.resnet101 import Resnet101


parser = argparse.ArgumentParser(description="Training FaceNet facial recognition model using center loss")
# Dataset
parser.add_argument('--dataroot', '-d', type=str, required=True, help="(REQUIRED) Absolute path to the dataset folder")
# LFW
parser.add_argument('--lfw', type=str, required=True, help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder")
parser.add_argument('--lfw_batch_size', default=32, type=int, help="Batch size for LFW dataset (default: 32)")
parser.add_argument('--lfw_validation_epoch', default=10, type=int, help="Perform LFW validation every n epoch (default: every 10 epochs)")
# Training settings
parser.add_argument('--model', type=str, default="resnet34", choices=["resnet34", "resnet50", "resnet101"],
    help="The required model architecture for training: ('resnet34', 'resnet50', 'resnet101'), (default: 'resnet34')"
)
parser.add_argument('--epochs', default=150, type=int, help="Required training epochs (default: 150)")
parser.add_argument('--batch_size', default=64, type=int, help="Batch size (default: 64)")
parser.add_argument('--num_workers', default=4, type=int, help="Number of workers for data loaders (default: 4)")
parser.add_argument('--valid_split', default=0.01, type=float, help="Validation dataset percentage to be used from the dataset (default: 0.01)")
parser.add_argument('--embedding_dim', default=128, type=int, help="Dimension of the embedding vector (default: 128)")
parser.add_argument('--pretrained', default=False, type=bool, help="Download a model pretrained on the ImageNet dataset (Default: False)")
parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate for Adam optimizer (default: 0.001)")
parser.add_argument('--center_loss_lr', default=0.5, type=float, help="Learning rate for center loss (default: 0.5)")
parser.add_argument('--center_loss_weight', default=0.003, type=float, help="Center loss weight (default: 0.003)")
args = parser.parse_args()


def main():
    dataroot = args.dataroot
    lfw_dataroot = args.lfw
    lfw_batch_size = args.lfw_batch_size
    lfw_validation_epoch = args.lfw_validation_epoch
    model_architecture = args.model
    batch_size = args.batch_size
    num_workers = args.num_workers
    validation_dataset_split_ratio = args.valid_split
    embedding_dimension = args.embedding_dim
    pretrained = args.pretrained
    learning_rate = args.learning_rate
    learning_rate_center_loss = args.center_loss_lr
    center_loss_weight = args.center_loss_weight

    # Define transforms for the training and validation sets
    data_transforms = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # Load the dataset
    dataset = torchvision.datasets.ImageFolder(
        root=dataroot,
        transform=data_transforms
    )

    # Subset the dataset into training and validation datasets
    num_classes = len(dataset.classes)
    print("Number of classes in dataset: {}".format(num_classes))
    num_validation = int(num_classes * validation_dataset_split_ratio)
    num_train = num_classes - num_validation
    indices = list(range(num_classes))
    np.random.seed(420)
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

    # Load the LFW dataset for validation
    lfw_loader = torch.utils.data.DataLoader(
        LFWDataset(
            dir=lfw_dataroot,
            pairs_path='LFW_pairs.txt',
            transform=data_transforms
        ),
        batch_size=lfw_batch_size, num_workers=num_workers
    )

    # Instantiate model
    if model_architecture=="resnet34":
        model = Resnet34(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture=="resnet50":
        model = Resnet50(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture=="resnet101":
        model = Resnet101(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )

    # Load model to GPU or multiple GPUs if available
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

    # Set loss functions
    criterion_crossentropy = nn.CrossEntropyLoss().cuda()
    criterion_centerloss = CenterLoss(num_classes=num_classes, feat_dim=embedding_dimension, use_gpu=True)

    # Optimize model parameters and the center loss object parameters
    parameters = list(model.parameters()) + list(criterion_centerloss.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)

    # Set learning rate reduction scheduler
    learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[40, 70],
        gamma=0.1
    )

    # Training loop
    epochs = args.epochs

    print("\nTraining starting for {} epochs:\n".format(epochs))
    total_time_start = time.time()

    for epoch in range(epochs):
        train_loss = 0
        validation_loss = 0

        epoch_time_start = time.time()
        # Training the model
        learning_rate_scheduler.step()
        model.train()
        progress_bar = tqdm(enumerate(train_dataloader))
        for batch_index, (data, labels) in progress_bar:
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
                # Remove the effect of center_loss_weight on updating centers
                parameter.grad.data *= (learning_rate_center_loss / (center_loss_weight * learning_rate))
            optimizer.step()
            # Update average training loss
            train_loss += loss.item()*data.size(0)

        # Validating the model
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            progress_bar = tqdm(enumerate(validation_dataloader))
            for batch_index, (data, labels) in progress_bar:
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
        print('Epoch {}:\tTraining Loss: {:.4f}\tValidation Loss: {:.4f}\tClassification Accuracy: {:.2f}%\t\
            Classification Error: {:.2f}%\tEpoch Time: {:.2f} minutes'.format(
            epoch+1, train_loss, validation_loss, accuracy, error, (epoch_time_end - epoch_time_start)/60
        ))

        # Validating on LFW dataset using KFold based on Euclidean distance metric
        if (epoch+1) % lfw_validation_epoch == 0 or (epoch+1) % epochs == 0:
            l2_distance = PairwiseDistance(2)
            distances, labels = [], []

            print("Validating on LFW! ...")
            progress_bar = tqdm(enumerate(lfw_loader))
            for batch_index, (data_a, data_b, label) in progress_bar:
                data_a, data_b, label = data_a.cuda(), data_b.cuda(), label.cuda()

                output_a, output_b = model(data_a), model(data_b)
                distance = l2_distance.forward(output_a, output_b)  # Euclidean distance
                distances.append(distance.cpu().detach().numpy())
                labels.append(label.cpu().detach().numpy())

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for distance in distances for subdist in distance])

            true_positive_rate, false_positive_rate, accuracy = evaluate_lfw(distances=distances, labels=labels)
            print("Accuracy on LFW: {:.4f}\t True Positive Rate: {:.4f}\t False Positive Rate: {:.4f}".format(
                np.mean(accuracy), np.mean(true_positive_rate), np.mean(false_positive_rate))
            )
            plot_roc_lfw(
                false_positive_rate=false_positive_rate, true_positive_rate=true_positive_rate, figure_name="roc.png"
            )

        # Save model checkpoint
        state = {
            'epoch': epoch,
            'num_classes': num_classes,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'validation_accuracy': accuracy,
            'validation_error': error,
            'elapsed_training_time_seconds': time.time() - total_time_start
        }
        # For storing data parallel model's state dictionary without 'module' parameter
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()

        torch.save(state, 'model_{}.pt'.format(model_architecture))

    # Training loop end
    total_time_end = time.time()
    total_time_elapsed = total_time_end - total_time_start

    print("\nTraining finished: total time elapsed: {:.2f} minutes.".format(total_time_elapsed/60))


if __name__ == '__main__':
    main()

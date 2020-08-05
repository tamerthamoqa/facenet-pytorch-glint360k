import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from torch.utils.data import DataLoader
from losses.center_loss import CenterLoss
from dataloaders.LFWDataset import LFWDataset
from validate_on_LFW import evaluate_lfw
from plot import plot_roc_lfw, plot_accuracy_lfw
from tqdm import tqdm
from models.resnet import Resnet18Center
from models.resnet import Resnet34Center
from models.resnet import Resnet50Center
from models.resnet import Resnet101Center
from models.resnet import Resnet152Center
from models.inceptionresnetv2 import InceptionResnetV2Center


parser = argparse.ArgumentParser(description="Training a facial recognition model using Cross Entropy Loss with Center Loss.")
# Dataset
parser.add_argument('--dataroot', '-d', type=str, required=True,
                    help="(REQUIRED) Absolute path to the dataset folder"
                    )
# LFW
parser.add_argument('--lfw', type=str, required=True,
                    help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder"
                    )
parser.add_argument('--lfw_batch_size', default=64, type=int,
                    help="Batch size for LFW dataset (default: 64)"
                    )
parser.add_argument('--lfw_validation_epoch_interval', default=1, type=int,
                    help="Perform LFW validation every n epoch interval (default: every 1 epoch)"
                    )
# Training settings
parser.add_argument('--model_architecture', type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "inceptionresnetv2"],
    help="The required model architecture for training: ('resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152', 'inceptionresnetv2'), (default: 'resnet34')"
                    )
parser.add_argument('--epochs', default=30, type=int,
                    help="Required training epochs (default: 30)"
                    )
parser.add_argument('--resume_path', default='',  type=str,
    help='path to latest model checkpoint: (Model_training_checkpoints/model_resnet34_epoch_1.pt file) (default: None)'
                    )
parser.add_argument('--batch_size', default=64, type=int,
                    help="Batch size (default: 64)"
                    )
parser.add_argument('--num_workers', default=8, type=int,
                    help="Number of workers for data loaders (default: 8)"
                    )
parser.add_argument('--embedding_dim', default=256, type=int,
                    help="Dimension of the embedding vector (default: 256)"
                    )
parser.add_argument('--pretrained', default=False, type=bool,
                    help="Download a model pretrained on the ImageNet dataset (Default: False)"
                    )
parser.add_argument('--optimizer', type=str, default="sgd", choices=["sgd", "adagrad", "rmsprop", "adam"],
    help="Required optimizer for training the model: ('sgd','adagrad','rmsprop','adam'), (default: 'sgd')"
                    )
parser.add_argument('--lr', default=0.001, type=float,
                    help="Learning rate for the optimizer (default: 0.001)"
                    )
parser.add_argument('--center_loss_lr', default=0.5, type=float,
                    help="Learning rate for center loss (default: 0.5)"
                    )
parser.add_argument('--center_loss_weight', default=0.007, type=float,
                    help="Center loss weight (default: 0.007)"
                    )
parser.add_argument('--image_size', default=224, type=int,
                    help='Input image size (default: 224 (224x224), must be 299x299 for Inception-ResNet-V2)'
                    )
args = parser.parse_args()


def set_model_architecture(model_architecture, pretrained, embedding_dimension, num_classes):
    if model_architecture == "resnet18":
        model = Resnet18Center(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet34":
        model = Resnet34Center(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet50":
        model = Resnet50Center(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet101":
        model = Resnet101Center(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet152":
        model = Resnet152Center(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "inceptionresnetv2":
        model = InceptionResnetV2Center(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    print("Using {} model architecture.".format(model_architecture))

    return model


def set_model_gpu_mode(model):
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

    return model, flag_train_multi_gpu


def set_optimizers(optimizer, model, learning_rate, learning_rate_center_loss, criterion_centerloss):
    if optimizer == "sgd":
        optimizer_model = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        optimizer_centerloss = optim.SGD(criterion_centerloss.parameters(), lr=learning_rate_center_loss, momentum=0.9)

    elif optimizer == "adagrad":
        optimizer_model = optim.Adagrad(model.parameters(), lr=learning_rate)
        optimizer_centerloss = optim.Adagrad(criterion_centerloss.parameters(), lr=learning_rate_center_loss)

    elif optimizer == "rmsprop":
        optimizer_model = optim.RMSprop(model.parameters(), lr=learning_rate)
        optimizer_centerloss = optim.RMSprop(criterion_centerloss.parameters(), lr=learning_rate_center_loss)

    elif optimizer == "adam":
        optimizer_model = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer_centerloss = optim.Adam(criterion_centerloss.parameters(), lr=learning_rate_center_loss)

    return optimizer_model, optimizer_centerloss


def validate_lfw(model, lfw_dataloader, model_architecture, epoch, epochs):
    model.eval()
    with torch.no_grad():
        l2_distance = PairwiseDistance(2).cuda()
        distances, labels = [], []

        print("Validating on LFW! ...")
        progress_bar = enumerate(tqdm(lfw_dataloader))

        for batch_index, (data_a, data_b, label) in progress_bar:
            data_a, data_b, label = data_a.cuda(), data_b.cuda(), label.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_lfw(
            distances=distances,
            labels=labels
        )

        # Print statistics and add to log
        print("Accuracy on LFW: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
              "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\t"
              "TAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
                np.mean(accuracy),
                np.std(accuracy),
                np.mean(precision),
                np.std(precision),
                np.mean(recall),
                np.std(recall),
                roc_auc,
                np.mean(best_distances),
                np.std(best_distances),
                np.mean(tar),
                np.std(tar),
                np.mean(far)
              )
        )
        with open('logs/lfw_{}_log_center.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch + 1,
                np.mean(accuracy),
                np.std(accuracy),
                np.mean(precision),
                np.std(precision),
                np.mean(recall),
                np.std(recall),
                roc_auc,
                np.mean(best_distances),
                np.std(best_distances),
                np.mean(tar)
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

    try:
        # Plot ROC curve
        plot_roc_lfw(
            false_positive_rate=false_positive_rate,
            true_positive_rate=true_positive_rate,
            figure_name="plots/roc_plots/roc_{}_epoch_{}_center.png".format(model_architecture, epoch + 1)
        )
        # Plot LFW accuracies plot
        plot_accuracy_lfw(
            log_dir="logs/lfw_{}_log_center.txt".format(model_architecture),
            epochs=epochs,
            figure_name="plots/lfw_accuracies_{}_center.png".format(model_architecture)
        )
    except Exception as e:
        print(e)

    return best_distances


def train_center(start_epoch, end_epoch, epochs, train_dataloader, lfw_dataloader, lfw_validation_epoch_interval,
                 model, model_architecture, criterion_crossentropy, criterion_centerloss, optimizer_model,
                 optimizer_centerloss, center_loss_weight, num_classes, embedding_dimension, batch_size,
                 flag_train_multi_gpu):

    for epoch in range(start_epoch, end_epoch):
        flag_validate_lfw = (epoch + 1) % lfw_validation_epoch_interval == 0 or (epoch + 1) % epochs == 0
        train_loss_sum = 0

        # Training the model
        model.train()
        progress_bar = enumerate(tqdm(train_dataloader))

        for batch_index, (data, labels) in progress_bar:
            data, labels = data.cuda(), labels.cuda()

            # Forward pass
            if flag_train_multi_gpu:
                embedding, logits = model.module.forward_training(data)
            else:
                embedding, logits = model.forward_training(data)

            # Calculate losses
            cross_entropy_loss = criterion_crossentropy(logits, labels)
            center_loss = criterion_centerloss(embedding, labels)
            loss = (center_loss * center_loss_weight) + cross_entropy_loss

            # Backward pass
            optimizer_model.zero_grad()
            optimizer_centerloss.zero_grad()
            loss.backward()
            optimizer_model.step()
            # Remove center_loss_weight impact on the learning of center vectors
            for param in criterion_centerloss.parameters():
                param.grad.data *= (1. / center_loss_weight)
            optimizer_centerloss.step()

            # Update training loss sum
            train_loss_sum += loss.item()*data.size(0)

        # Calculate average losses in epoch
        avg_train_loss = train_loss_sum / len(train_dataloader.dataset)

        # Print training and validation statistics and add to log
        print('Epoch {}:\t Average Training Loss: {:.4f}'.format(
                epoch + 1,
                avg_train_loss
            )
        )
        with open('logs/{}_log_center.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch + 1,
                avg_train_loss
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

        # Validating on LFW dataset using KFold based on Euclidean distance metric
        if flag_validate_lfw:
            best_distances = validate_lfw(
                model=model,
                lfw_dataloader=lfw_dataloader,
                model_architecture=model_architecture,
                epoch=epoch,
                epochs=epochs
            )

        # Save model checkpoint
        state = {
            'epoch': epoch + 1,
            'num_classes': num_classes,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer_model.state_dict(),
            'optimizer_centerloss_state_dict': optimizer_centerloss.state_dict()
        }

        # For storing data parallel model's state dictionary without 'module' parameter
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()

        # For storing best euclidean distance threshold during LFW validation
        if flag_validate_lfw:
            state['best_distance_threshold'] = np.mean(best_distances)

        # Save model checkpoint
        torch.save(state, 'model_training_checkpoints/model_{}_center_epoch_{}.pt'.format(
                model_architecture,
                epoch + 1
            )
        )


def main():
    dataroot = args.dataroot
    lfw_dataroot = args.lfw
    lfw_batch_size = args.lfw_batch_size
    lfw_validation_epoch_interval = args.lfw_validation_epoch_interval
    model_architecture = args.model_architecture
    epochs = args.epochs
    resume_path = args.resume_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    embedding_dimension = args.embedding_dim
    pretrained = args.pretrained
    optimizer = args.optimizer
    learning_rate = args.lr
    learning_rate_center_loss = args.center_loss_lr
    center_loss_weight = args.center_loss_weight
    image_size = args.image_size
    start_epoch = 0

    # Define image data pre-processing transforms
    #   ToTensor() normalizes pixel values between [0, 1]
    #   Normalize(mean=[0.5157, 0.4062, 0.3550], std=[0.2858, 0.2515, 0.2433]) normalizes pixel values to be mean
    #    of zero and standard deviation of 1 according to the calculated VGGFace2 with cropped faces dataset RGB
    #    channels' mean and std values by calculate_vggface2_rgb_mean_std.py in 'datasets' folder.
    data_transforms = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5157, 0.4062, 0.3550],
            std=[0.2858, 0.2515, 0.2433]
        )
    ])

    lfw_transforms = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5157, 0.4062, 0.3550],
            std=[0.2858, 0.2515, 0.2433]
        )
    ])

    # Load the dataset
    dataset = torchvision.datasets.ImageFolder(
        root=dataroot,
        transform=data_transforms
    )
    num_classes = len(dataset.classes)
    print("Number of classes in training dataset: {}".format(num_classes))

    # Define the dataloaders
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    lfw_dataloader = torch.utils.data.DataLoader(
        dataset=LFWDataset(
            dir=lfw_dataroot,
            pairs_path='datasets/LFW_pairs.txt',
            transform=lfw_transforms
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    # Instantiate model
    model = set_model_architecture(
        model_architecture=model_architecture,
        pretrained=pretrained,
        embedding_dimension=embedding_dimension,
        num_classes=num_classes
    )

    # Load model to GPU or multiple GPUs if available
    model, flag_train_multi_gpu = set_model_gpu_mode(model)

    # Set loss functions
    criterion_crossentropy = nn.CrossEntropyLoss()
    criterion_centerloss = CenterLoss(num_classes=num_classes, feat_dim=embedding_dimension)

    # Set optimizers
    optimizer_model, optimizer_centerloss = set_optimizers(
        optimizer=optimizer,
        model=model,
        learning_rate=learning_rate,
        learning_rate_center_loss=learning_rate_center_loss,
        criterion_centerloss=criterion_centerloss
    )

    # Resume from a model checkpoint
    if resume_path:
        if os.path.isfile(resume_path):
            print("Loading checkpoint {} ...".format(resume_path))

            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
            if flag_train_multi_gpu:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
            optimizer_centerloss.load_state_dict(checkpoint['optimizer_centerloss_state_dict'])

            print("Checkpoint loaded: start epoch from checkpoint = {}\nRunning for {} epochs.\n".format(
                    start_epoch,
                    epochs - start_epoch
                )
            )
        else:
            print("WARNING: No checkpoint found at {}!\nTraining from scratch.".format(resume_path))

    # Start Training loop
    print("Training using cross entropy loss with center loss starting for {} epochs:\n".format(epochs - start_epoch))
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs

    # Start training model using Cross Entropy Loss with Center Loss
    train_center(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        epochs=epochs,
        train_dataloader=train_dataloader,
        lfw_dataloader=lfw_dataloader,
        lfw_validation_epoch_interval=lfw_validation_epoch_interval,
        model=model,
        model_architecture=model_architecture,
        criterion_crossentropy=criterion_crossentropy,
        criterion_centerloss=criterion_centerloss,
        optimizer_model=optimizer_model,
        optimizer_centerloss=optimizer_centerloss,
        center_loss_weight=center_loss_weight,
        num_classes=num_classes,
        embedding_dimension=embedding_dimension,
        batch_size=batch_size,
        flag_train_multi_gpu=flag_train_multi_gpu
    )


if __name__ == '__main__':
    main()

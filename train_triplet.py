import numpy as np
import argparse
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from dataloaders.LFWDataset import LFWDataset
from losses.triplet_loss import TripletLoss
from dataloaders.triplet_loss_dataloader import TripletFaceDataset
from validate_on_LFW import evaluate_lfw
from plot import plot_roc_lfw, plot_accuracy_lfw, plot_triplet_losses
from tqdm import tqdm
from models.resnet import Resnet18Triplet
from models.resnet import Resnet34Triplet
from models.resnet import Resnet50Triplet
from models.resnet import Resnet101Triplet
from models.resnet import Resnet152Triplet
from models.inceptionresnetv2 import InceptionResnetV2Triplet
from models.mobilenetv2 import MobileNetV2Triplet


parser = argparse.ArgumentParser(description="Training a FaceNet facial recognition model using Triplet Loss.")
# Dataset
parser.add_argument('--dataroot', '-d', type=str, required=True,
                    help="(REQUIRED) Absolute path to the dataset folder"
                    )
# LFW
parser.add_argument('--lfw', type=str, required=True,
                    help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder"
                    )
parser.add_argument('--dataset_csv', type=str, default='datasets/vggface2_full.csv',
                    help="Path to the csv file containing the image paths of the training dataset."
                    )
parser.add_argument('--lfw_batch_size', default=256, type=int,
                    help="Batch size for LFW dataset (default: 256)"
                    )
parser.add_argument('--lfw_validation_epoch_interval', default=1, type=int,
                    help="Perform LFW validation every n epoch interval (default: every 1 epoch)"
                    )
# Training settings
parser.add_argument('--model_architecture', type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "inceptionresnetv2", "mobilenetv2"],
    help="The required model architecture for training: ('resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152', 'inceptionresnetv2', 'mobilenetv2'), (default: 'resnet18')"
                    )
parser.add_argument('--epochs', default=50, type=int,
                    help="Required training epochs (default: 50)"
                    )
parser.add_argument('--training_triplets_path', default=None, type=str,
    help="Path to training triplets numpy file in 'datasets/' folder to skip training triplet generation step."
                    )
parser.add_argument('--num_triplets_train', default=1100000, type=int,
                    help="Number of triplets for training (default: 1100000)"
                    )
parser.add_argument('--resume_path', default='',  type=str,
    help='path to latest model checkpoint: (model_training_checkpoints/model_resnet18_epoch_1.pt file) (default: None)'
                    )
parser.add_argument('--batch_size', default=256, type=int,
                    help="Batch size (default: 256)"
                    )
parser.add_argument('--num_workers', default=1, type=int,
                    help="Number of workers for data loaders (default: 1)"
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
parser.add_argument('--lr', default=0.1, type=float,
                    help="Learning rate for the optimizer (default: 0.1)"
                    )
parser.add_argument('--margin', default=0.2, type=float,
                    help='margin for triplet loss (default: 0.2)'
                    )
parser.add_argument('--image_size', default=224, type=int,
                    help='Input image size (default: 224 (224x224), must be 299x299 for Inception-ResNet-V2)'
                    )
args = parser.parse_args()


def set_model_architecture(model_architecture, pretrained, embedding_dimension):
    if model_architecture == "resnet18":
        model = Resnet18Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet34":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet50":
        model = Resnet50Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet101":
        model = Resnet101Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet152":
        model = Resnet152Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "inceptionresnetv2":
        model = InceptionResnetV2Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "mobilenetv2":
        model = MobileNetV2Triplet(
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


def set_optimizer(optimizer, model, learning_rate):
    if optimizer == "sgd":
        optimizer_model = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    elif optimizer == "adagrad":
        optimizer_model = optim.Adagrad(model.parameters(), lr=learning_rate)

    elif optimizer == "rmsprop":
        optimizer_model = optim.RMSprop(model.parameters(), lr=learning_rate)

    elif optimizer == "adam":
        optimizer_model = optim.Adam(model.parameters(), lr=learning_rate)

    return optimizer_model


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
        with open('logs/lfw_{}_log_triplet.txt'.format(model_architecture), 'a') as f:
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
            figure_name="plots/roc_plots/roc_{}_epoch_{}_triplet.png".format(model_architecture, epoch + 1)
        )
        # Plot LFW accuracies plot
        plot_accuracy_lfw(
            log_dir="logs/lfw_{}_log_triplet.txt".format(model_architecture),
            epochs=epochs,
            figure_name="plots/lfw_accuracies_{}_triplet.png".format(model_architecture)
        )
    except Exception as e:
        print(e)

    return best_distances


def forward_pass(anc_imgs, pos_imgs, neg_imgs, model, optimizer_model, batch_idx, use_cpu=False):
    # If CUDA is Out of Memory, load model to cpu and do a forward pass
    if use_cpu:
        gc.collect()
        torch.cuda.empty_cache()

        # 1- Anchors
        anc_imgs = anc_imgs.cpu()
        anc_embeddings = model(anc_imgs)
        del anc_imgs
        gc.collect()

        # 2- Positives
        pos_imgs = pos_imgs.cpu()
        pos_embeddings = model(pos_imgs)
        del pos_imgs
        gc.collect()

        # 3- Negatives
        neg_imgs = neg_imgs.cpu()
        neg_embeddings = model(neg_imgs)
        del neg_imgs
        gc.collect()

        return anc_embeddings, pos_embeddings, neg_embeddings, model, optimizer_model

    # Forward pass on CUDA
    else:
        try:
            # Model is already loaded to cuda
            # 1- Anchors
            anc_imgs = anc_imgs.cuda()
            anc_embeddings = model(anc_imgs)
            anc_imgs = anc_imgs.cpu()
            anc_embeddings = anc_embeddings.cpu()

            # 2- Positives
            pos_imgs = pos_imgs.cuda()
            pos_embeddings = model(pos_imgs)
            pos_imgs = pos_imgs.cpu()
            pos_embeddings = pos_embeddings.cpu()

            # 3- Negatives
            neg_imgs = neg_imgs.cuda()
            neg_embeddings = model(neg_imgs)
            neg_imgs = neg_imgs.cpu()
            neg_embeddings = neg_embeddings.cpu()

            del anc_imgs, pos_imgs, neg_imgs
            gc.collect()

            return anc_embeddings, pos_embeddings, neg_embeddings, model, optimizer_model

        # CUDA Out of Memory Exception Handling
        except RuntimeError as e:
            # Inspired by:
            # https://github.com/pytorch/fairseq/blob/50a671f78d0c8de0392f924180db72ac9b41b801/fairseq/trainer.py#L284
            if "out of memory" in str(e):
                print("CUDA Out of Memory at iteration {}. Retrying iteration on CPU!".format(batch_idx))
                model = model.cpu()
                # Copied from https://github.com/pytorch/pytorch/issues/2830#issuecomment-336031198
                # No optimizer.cpu() available, this is the way to make an optimizer loaded with cuda tensors load
                #  with cpu tensors
                for state in optimizer_model.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()

                gc.collect()
                torch.cuda.empty_cache()

                return forward_pass(
                    anc_imgs=anc_imgs,
                    pos_imgs=pos_imgs,
                    neg_imgs=neg_imgs,
                    model=model,
                    optimizer_model=optimizer_model,
                    batch_idx=batch_idx,
                    use_cpu=True
                )
            else:
                raise e


def train_triplet(start_epoch, end_epoch, epochs, train_dataloader, lfw_dataloader, lfw_validation_epoch_interval,
                  model, model_architecture, optimizer_model, embedding_dimension, batch_size, margin,
                  flag_train_multi_gpu):

    for epoch in range(start_epoch, end_epoch):
        flag_validate_lfw = (epoch + 1) % lfw_validation_epoch_interval == 0 or (epoch + 1) % epochs == 0
        triplet_loss_sum = 0
        num_valid_training_triplets = 0
        l2_distance = PairwiseDistance(2)

        # Training pass
        model.train()
        progress_bar = enumerate(tqdm(train_dataloader))

        for batch_idx, (batch_sample) in progress_bar:
            # Make sure model and optimizer are loaded to cuda first, when an Out of Memory Exception occurs,
            #  continue on cpu for the rest of the iteration in forward_pass() to avoid the following error on
            #  optimizer_model.step():
            #   RuntimeError: Expected all tensors to be on the same device, but found at least two devices,
            #    cuda:0 and cpu!
            model = model.cuda()
            for state in optimizer_model.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            # Forward pass - compute embeddings
            anc_imgs = batch_sample['anc_img']
            pos_imgs = batch_sample['pos_img']
            neg_imgs = batch_sample['neg_img']

            anc_embeddings, pos_embeddings, neg_embeddings, model, optimizer_model = forward_pass(
                anc_imgs=anc_imgs,
                pos_imgs=pos_imgs,
                neg_imgs=neg_imgs,
                model=model,
                optimizer_model=optimizer_model,
                batch_idx=batch_idx,
                use_cpu=False
            )

            # Forward pass - choose hard negatives only for training
            pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
            neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)

            all = (neg_dists - pos_dists < margin).cpu().numpy().flatten()

            hard_triplets = np.where(all == 1)
            if len(hard_triplets[0]) == 0:
                continue

            anc_hard_embeddings = anc_embeddings[hard_triplets]
            pos_hard_embeddings = pos_embeddings[hard_triplets]
            neg_hard_embeddings = neg_embeddings[hard_triplets]

            # Calculate triplet loss
            triplet_loss = TripletLoss(margin=margin).forward(
                anchor=anc_hard_embeddings,
                positive=pos_hard_embeddings,
                negative=neg_hard_embeddings
            )

            # Calculating loss
            triplet_loss_sum += triplet_loss.item()
            num_valid_training_triplets += len(anc_hard_embeddings)

            # Backward pass
            optimizer_model.zero_grad()
            triplet_loss.backward()
            optimizer_model.step()

        # Model only trains on hard negative triplets
        avg_triplet_loss = 0 if (num_valid_training_triplets == 0) else triplet_loss_sum / num_valid_training_triplets

        # Print training statistics and add to log
        print('Epoch {}:\tAverage Triplet Loss: {:.4f}\tNumber of valid training triplets in epoch: {}'.format(
                epoch + 1,
                avg_triplet_loss,
                num_valid_training_triplets
            )
        )
        with open('logs/{}_log_triplet.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch + 1,
                avg_triplet_loss,
                num_valid_training_triplets
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

        try:
            # Plot Triplet losses plot
            plot_triplet_losses(
                log_dir="logs/{}_log_triplet.txt".format(model_architecture),
                epochs=epochs,
                figure_name="plots/triplet_losses_{}.png".format(model_architecture)
            )
        except Exception as e:
            print(e)

        # Evaluation pass on LFW dataset
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
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer_model.state_dict()
        }

        # For storing data parallel model's state dictionary without 'module' parameter
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()

        # For storing best euclidean distance threshold during LFW validation
        if flag_validate_lfw:
            state['best_distance_threshold'] = np.mean(best_distances)

        # Save model checkpoint
        torch.save(state, 'model_training_checkpoints/model_{}_triplet_epoch_{}.pt'.format(
                model_architecture,
                epoch + 1
            )
        )


def main():
    dataroot = args.dataroot
    lfw_dataroot = args.lfw
    dataset_csv = args.dataset_csv
    lfw_batch_size = args.lfw_batch_size
    lfw_validation_epoch_interval = args.lfw_validation_epoch_interval
    model_architecture = args.model_architecture
    epochs = args.epochs
    training_triplets_path = args.training_triplets_path
    num_triplets_train = args.num_triplets_train
    resume_path = args.resume_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    embedding_dimension = args.embedding_dim
    pretrained = args.pretrained
    optimizer = args.optimizer
    learning_rate = args.lr
    margin = args.margin
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

    # Set dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset=TripletFaceDataset(
            root_dir=dataroot,
            csv_name=dataset_csv,
            num_triplets=num_triplets_train,
            training_triplets_path=training_triplets_path,
            transform=data_transforms
        ),
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
        embedding_dimension=embedding_dimension
    )

    # Load model to GPU or multiple GPUs if available
    model, flag_train_multi_gpu = set_model_gpu_mode(model)

    # Set optimizer
    optimizer_model = set_optimizer(
        optimizer=optimizer,
        model=model,
        learning_rate=learning_rate
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

            print("Checkpoint loaded: start epoch from checkpoint = {}\nRunning for {} epochs.\n".format(
                    start_epoch,
                    epochs - start_epoch
                )
            )
        else:
            print("WARNING: No checkpoint found at {}!\nTraining from scratch.".format(resume_path))

    # Start Training loop
    print("Training using triplet loss on {} triplets starting for {} epochs:\n".format(
            num_triplets_train,
            epochs - start_epoch
        )
    )

    start_epoch = start_epoch
    end_epoch = start_epoch + epochs

    # Start training model using Triplet Loss
    train_triplet(
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        epochs=epochs,
        train_dataloader=train_dataloader,
        lfw_dataloader=lfw_dataloader,
        lfw_validation_epoch_interval=lfw_validation_epoch_interval,
        model=model,
        model_architecture=model_architecture,
        optimizer_model=optimizer_model,
        embedding_dimension=embedding_dimension,
        batch_size=batch_size,
        margin=margin,
        flag_train_multi_gpu=flag_train_multi_gpu
    )


if __name__ == '__main__':
    main()

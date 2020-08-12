import torch.nn as nn
from torch.nn import functional as F
from .utils_resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class Resnet18Center(nn.Module):
    """Constructs a ResNet-18 model for FaceNet training using center loss with cross entropy loss.

    Args:
        num_classes (int): Number of classes in the training dataset required for cross entropy loss.
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using center loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """
    def __init__(self, num_classes, embedding_dimension=256, pretrained=False):
        super(Resnet18Center, self).__init__()
        self.model = resnet18(pretrained=pretrained)

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )
        # Output logits for cross entropy loss
        self.model.classifier = nn.Linear(embedding_dimension, num_classes)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization ."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward_training(self, images):
        """Forward pass during training to output both the l2-normed embedding vector and logits
          for cross entropy loss and center loss.
        """
        embedding = self.forward(images)
        logits = self.model.classifier(embedding)

        return embedding, logits


class Resnet18Triplet(nn.Module):
    """Constructs a ResNet-18 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=256, pretrained=False):
        super(Resnet18Triplet, self).__init__()
        self.model = resnet18(pretrained=pretrained)

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class Resnet34Center(nn.Module):
    """Constructs a ResNet-34 model for FaceNet training using center loss with cross entropy loss.

    Args:
        num_classes (int): Number of classes in the training dataset required for cross entropy loss.
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using center loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """
    def __init__(self, num_classes, embedding_dimension=256, pretrained=False):
        super(Resnet34Center, self).__init__()
        self.model = resnet34(pretrained=pretrained)

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )
        # Output logits for cross entropy loss
        self.model.classifier = nn.Linear(embedding_dimension, num_classes)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward_training(self, images):
        """Forward pass during training to output both the l2-normed embedding vector and logits
          for cross entropy loss and center loss.
        """
        embedding = self.forward(images)
        logits = self.model.classifier(embedding)

        return embedding, logits


class Resnet34Triplet(nn.Module):
    """Constructs a ResNet-34 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=256, pretrained=False):
        super(Resnet34Triplet, self).__init__()
        self.model = resnet34(pretrained=pretrained)

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class Resnet50Center(nn.Module):
    """Constructs a ResNet-50 model for FaceNet training using center loss with cross entropy loss.

    Args:
        num_classes (int): Number of classes in the training dataset required for cross entropy loss.
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using center loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """
    def __init__(self, num_classes, embedding_dimension=256, pretrained=False):
        super(Resnet50Center, self).__init__()
        self.model = resnet50(pretrained=pretrained)

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )
        # Output logits for cross entropy loss
        self.model.classifier = nn.Linear(embedding_dimension, num_classes)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward_training(self, images):
        """Forward pass during training to output both the l2-normed embedding vector and logits
          for cross entropy loss and center loss.
        """
        embedding = self.forward(images)
        logits = self.model.classifier(embedding)

        return embedding, logits


class Resnet50Triplet(nn.Module):
    """Constructs a ResNet-50 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=256, pretrained=False):
        super(Resnet50Triplet, self).__init__()
        self.model = resnet50(pretrained=pretrained)

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class Resnet101Center(nn.Module):
    """Constructs a ResNet-101 model for FaceNet training using center loss with cross entropy loss.

    Args:
        num_classes (int): Number of classes in the training dataset required for cross entropy loss.
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using center loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """
    def __init__(self, num_classes, embedding_dimension=256, pretrained=False):
        super(Resnet101Center, self).__init__()
        self.model = resnet101(pretrained=pretrained)

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )
        # Output logits for cross entropy loss
        self.model.classifier = nn.Linear(embedding_dimension, num_classes)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward_training(self, images):
        """Forward pass during training to output both the l2-normed embedding vector and logits
          for cross entropy loss and center loss.
        """
        embedding = self.forward(images)
        logits = self.model.classifier(embedding)

        return embedding, logits


class Resnet101Triplet(nn.Module):
    """Constructs a ResNet-101 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=256, pretrained=False):
        super(Resnet101Triplet, self).__init__()
        self.model = resnet101(pretrained=pretrained)

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class Resnet152Center(nn.Module):
    """Constructs a ResNet-152 model for FaceNet training using center loss with cross entropy loss.

    Args:
        num_classes (int): Number of classes in the training dataset required for cross entropy loss.
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using center loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """
    def __init__(self, num_classes, embedding_dimension=256, pretrained=False):
        super(Resnet152Center, self).__init__()
        self.model = resnet152(pretrained=pretrained)

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )
        # Output logits for cross entropy loss
        self.model.classifier = nn.Linear(embedding_dimension, num_classes)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    def forward_training(self, images):
        """Forward pass during training to output both the l2-normed embedding vector and logits
          for cross entropy loss and center loss.
        """
        embedding = self.forward(images)
        logits = self.model.classifier(embedding)

        return embedding, logits


class Resnet152Triplet(nn.Module):
    """Constructs a ResNet-152 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using triplet loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """

    def __init__(self, embedding_dimension=256, pretrained=False):
        super(Resnet152Triplet, self).__init__()
        self.model = resnet152(pretrained=pretrained)

        # Output embedding
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(input_features_fc_layer, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

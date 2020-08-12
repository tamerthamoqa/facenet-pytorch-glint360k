import torch.nn as nn
from torch.nn import functional as F
from .utils_inceptionresnetv2 import inceptionresnetv2


class InceptionResnetV2Center(nn.Module):
    """Constructs an Inception-ResNet-V2 model for FaceNet training using center loss with cross entropy loss.

    Args:
        num_classes (int): Number of classes in the training dataset required for cross entropy loss.
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                    using center loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                            Defaults to False.
    """
    def __init__(self, num_classes, embedding_dimension=256, pretrained=False):
        super(InceptionResnetV2Center, self).__init__()
        if pretrained:
            self.model = inceptionresnetv2(pretrained='imagenet')
        else:
            self.model = inceptionresnetv2(pretrained=pretrained)

        # Output embedding
        self.model.last_linear = nn.Sequential(
            nn.Linear(1536, embedding_dimension, bias=False),
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


class InceptionResnetV2Triplet(nn.Module):
    """Constructs an Inception-ResNet-V2 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                    using triplet loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                            Defaults to False.
    """
    def __init__(self, embedding_dimension=256, pretrained=False):
        super(InceptionResnetV2Triplet, self).__init__()
        if pretrained:
            self.model = inceptionresnetv2(pretrained='imagenet')
        else:
            self.model = inceptionresnetv2(pretrained=pretrained)

        # Output embedding
        self.model.last_linear = nn.Sequential(
            nn.Linear(1536, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

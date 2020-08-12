import torch.nn as nn
from torch.nn import functional as F
from .utils_mobilenetv2 import mobilenet_v2


class MobileNetV2Center(nn.Module):
    """Constructs a MobileNet-V2 model for FaceNet training using center loss with cross entropy loss.

    Args:
        num_classes (int): Number of classes in the training dataset required for cross entropy loss.
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                    using center loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                            Defaults to False.
    """
    def __init__(self, num_classes, embedding_dimension=256, pretrained=False):
        super(MobileNetV2Center, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)

        # Output embedding
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )
        # Output logits for cross entropy loss
        # The last sequential layer of the original mobilenet-v2 model is 'classifier' hence the classifier2 naming
        self.model.classifier2 = nn.Linear(embedding_dimension, num_classes)

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
        logits = self.model.classifier2(embedding)

        return embedding, logits


class MobileNetV2Triplet(nn.Module):
    """Constructs a MobileNet-V2 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                    using triplet loss. Defaults to 256.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                            Defaults to False.
    """
    def __init__(self, embedding_dimension=256, pretrained=False):
        super(MobileNetV2Triplet, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)

        # Output embedding
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension, eps=0.001, momentum=0.1, affine=True)
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

import torch
import torch.nn as nn
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
        #  Based on https://arxiv.org/abs/1703.07737
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dimension),
            nn.BatchNorm1d(embedding_dimension)
        )
        # Output logits for cross entropy loss
        # The last sequential layer of the original mobilenet-v2 model is 'classifier' hence the classifier2 naming
        self.model.classifier2 = nn.Linear(embedding_dimension, num_classes)

    def l2_norm(self, input):
        """Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
        """
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)

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
        #  Based on https://arxiv.org/abs/1703.07737
        self.model.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dimension),
            nn.BatchNorm1d(embedding_dimension)
        )

    def l2_norm(self, input):
        """Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
        """
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)

        return embedding

import torch
import torch.nn as nn
import torchvision.models as models


class Resnet101(nn.Module):
    """Constructs a ResNet-101 model for FaceNet training using center loss with cross entropy loss.

    Args:
        num_classes (int): Number of classes in the training dataset required for cross entropy loss.
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                   using center loss. Defaults to 128.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                           Defaults to False.
    """
    def __init__(self, num_classes, embedding_dimension=128, pretrained=False):
        super(Resnet101, self).__init__()
        self.model = models.resnet101(pretrained=pretrained)
        self.embedding_dimension = embedding_dimension
        input_features_fc_layer = self.model.fc.in_features
        # Output embedding
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)
        # Output logits for cross entropy loss
        self.model.classifier = nn.Linear(embedding_dimension, num_classes)

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
        """Forward pass to output the embedding vector (feature vector) after l2-normalization"""
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)

        return embedding

    def forward_training(self, images):
        """Forward pass during training to output both the embedding vector l2 norm and logits
          for cross entropy loss and center loss.
        """
        embedding = self.forward(images)
        logits = self.model.classifier(embedding)

        return embedding, logits

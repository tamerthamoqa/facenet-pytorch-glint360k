# facenet-pytorch-centerloss-vggface2 (IN PROGRESS)
A PyTorch implementation  of the [FaceNet](https://arxiv.org/abs/1503.03832)[1] paper for facial recognition using Cross Entropy Loss with [Center Loss](https://ydwen.github.io/papers/WenECCV16.pdf)[2] the implementation of which is imported from KaiyangZhou's 'pytorch-center-loss' [repository](https://github.com/KaiyangZhou/pytorch-center-loss). Training is done on the [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)[3] dataset containing 3.3 million face images based on over 9000 human identities.
## Steps
1. Download the VGGFace2 [dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/).
2. Download the Labeled Faces in the Wild [dataset](http://vis-www.cs.umass.edu/lfw/#download).  
3. For face alignment for both VGGFace2 and LFW datasets I used David Sandberg's face alignment script via MTCNN (Multi-task Cascaded Convolutional Neural Networks) from his 'facenet' [repository](https://github.com/davidsandberg/facenet):
 Steps to follow [here](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1#face-alignment) and [here](https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW#4-align-the-lfw-dataset).
4.  Type in ```python train.py -h``` to see the list of options of training. 
 &nbsp;

    __Note:__ '--dataroot' and '--lfw' arguments are required!

5. Run ```python train.py --dataroot "absolute path to VGGFace2 dataset folder" --lfw "absolute path to LFW dataset folder"```    
```
usage: train.py [-h] --dataroot DATAROOT --lfw LFW
                [--lfw_batch_size LFW_BATCH_SIZE]
                [--lfw_validation_epoch_interval LFW_VALIDATION_EPOCH_INTERVAL]
                [--model {resnet34,resnet50,resnet101}] [--epochs EPOCHS]
                [--resume_path RESUME_PATH] [--batch_size BATCH_SIZE]
                [--num_workers NUM_WORKERS] [--valid_split VALID_SPLIT]
                [--embedding_dim EMBEDDING_DIM] [--pretrained PRETRAINED]
                [--lr LR] [--center_loss_lr CENTER_LOSS_LR]
                [--center_loss_weight CENTER_LOSS_WEIGHT]

Training FaceNet facial recognition model using center loss

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT, -d DATAROOT
                        (REQUIRED) Absolute path to the dataset folder
  --lfw LFW             (REQUIRED) Absolute path to the labeled faces in the
                        wild dataset folder
  --lfw_batch_size LFW_BATCH_SIZE
                        Batch size for LFW dataset (default: 12)
  --lfw_validation_epoch_interval LFW_VALIDATION_EPOCH_INTERVAL
                        Perform LFW validation every n epoch interval
                        (default: every 5 epochs)
  --model {resnet34,resnet50,resnet101}
                        The required model architecture for training:
                        ('resnet34', 'resnet50', 'resnet101'), (default:
                        'resnet34')
  --epochs EPOCHS       Required training epochs (default: 275)
  --resume_path RESUME_PATH
                        path to latest model checkpoint (default: None)
  --batch_size BATCH_SIZE
                        Batch size (default: 64)
  --num_workers NUM_WORKERS
                        Number of workers for data loaders (default: 4)
  --valid_split VALID_SPLIT
                        Validation dataset percentage to be used from the
                        dataset (default: 0.01)
  --embedding_dim EMBEDDING_DIM
                        Dimension of the embedding vector (default: 128)
  --pretrained PRETRAINED
                        Download a model pretrained on the ImageNet dataset
                        (Default: False)
  --lr LR               Learning rate for the model using Adam optimizer
                        (default: 0.05)
  --center_loss_lr CENTER_LOSS_LR
                        Learning rate for center loss using Adam optimizer
                        (default: 0.5)
  --center_loss_weight CENTER_LOSS_WEIGHT
                        Center loss weight (default: 0.003)
```

## Model state dictionary
```
state = {
            'epoch': epoch+1,
            'num_classes': num_classes,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer_model.state_dict(),
            'optimizer_centerloss_state_dict': optimizer_centerloss.state_dict(),
            'learning_rate_scheduler_state_dict': learning_rate_scheduler.state_dict()            
        }
``` 

## Further work
1. Train and share models with performance metrics (Resnet-34, Resnet-50, Resnet-101).
2. Add performance metrics on the Labeled Faces in the Wild[4] [dataset](http://vis-www.cs.umass.edu/lfw/) for the trained models.

## References
* [1] Florian Schroff, Dmitry Kalenichenko, James Philbin, “FaceNet: A Unified Embedding for Face Recognition and Clustering”:
 [paper](https://arxiv.org/abs/1503.03832)

* [2] Yandong Wen, Kaipeng Zhang, Zhifeng Li, Yu Qiao, "A Discriminative Feature Learning Approachfor Deep Face Recognition": [paper](https://ydwen.github.io/papers/WenECCV16.pdf)

* [3] Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman
"VGGFace2: A dataset for recognising faces across pose and age":
[paper](https://arxiv.org/abs/1710.08092), [dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)

* [4] Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller.
"Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments": [paper](http://vis-www.cs.umass.edu/lfw/lfw.pdf)

## Inspirations (repositories)
* https://github.com/davidsandberg/facenet
* https://github.com/liorshk/facenet_pytorch
* https://github.com/KaiyangZhou/pytorch-center-loss (imported as center_loss.py)

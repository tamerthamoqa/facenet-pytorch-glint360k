import argparse
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from tqdm import tqdm
sys.path.append("..")
from models.resnet import Resnet18Triplet
from datasets.LFWDataset import LFWDataset
from validate_on_LFW import evaluate_lfw


from aligner.transform_align import transform_align # if you use: from YourClassParentDir.YourClass import YourClass   # means Class YourClass
# or use YourClass.YourClass(), it works

landmark_predictor_weights = './aligner/data_weights/shape_predictor_68_face_landmarks.dat'
face_detector_weights_path = './aligner/data_weights'

parser = argparse.ArgumentParser("Tests a model and prints its True Acceptance Rate at a specified False Acceptance Rate on the Labeled Faces in the Wild Dataset.")
parser.add_argument('--lfw', type=str, required=True,
                    help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder (raw images)"
                    )
parser.add_argument('--model_path', type=str, required=True,
                    help='path to model checkpoint: (model_training_checkpoints/model_resnet18_epoch_1.pt file) (default: None)'
                    )
parser.add_argument('--far_target', default=1e-3,  type=float,
                    help='The False Accept Rate to calculate the True Acceptance Rate (TAR) at, (default: 1e-3).'
                    )
parser.add_argument("--batch_size", default=200,  type=int, 
                    help="batch size, default 200, lower values reduce VRAM requirement and speed"
                   )
args = parser.parse_args()


def main():
    lfw_dataroot = args.lfw
    model_path = args.model_path
    far_target = args.far_target
    batch_size = args.batch_size

    flag_gpu_available = torch.cuda.is_available()

    if flag_gpu_available:
        device = torch.device("cuda")
        print('Using GPU')
    else:
        device = torch.device("cpu")
        print('Using CPU')
        
    checkpoint = torch.load(model_path, map_location=device)
    model = Resnet18Triplet(embedding_dimension=checkpoint['embedding_dimension'])
    model.load_state_dict(checkpoint['model_state_dict'])

# desiredFaceWidth, height and desiredLeftEye work together to produce centered aligned faces
# desiredLeftEye (x,y) says how the face moves in the heightxwidth window. bigger y moves the face down
# bigger x zooms-out the face and severe values migh flip the face upside down, you wan it in range [0.2;0.36]
    desiredFaceHeight = 448 # set non-equal width, height so the tf_resize stretch the faces, select bigger than 224 values so we don't lose too much of the image quality
    desiredFaceWidth = 352
    desiredLeftEye=(0.28, 0.35) 

    tf_align = transform_align(landmark_predictor_weight_path=landmark_predictor_weights, face_detector_path=face_detector_weights_path,
                           desiredFaceWidth=desiredFaceWidth, desiredFaceHeight=desiredFaceHeight,
                           desiredLeftEye=desiredLeftEye)

    lfw_transforms = transforms.Compose([
        tf_align,
        transforms.Resize(size=(140, 140)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6071, 0.4609, 0.3944],
            std=[0.2457, 0.2175, 0.2129]
        )
    ])

    lfw_dataloader = torch.utils.data.DataLoader(
        dataset=LFWDataset(
            dir=lfw_dataroot,
            pairs_path='../datasets/LFW_pairs.txt',
            transform=lfw_transforms
        ),
        batch_size=batch_size, # default = 256; 160 - allows running under 2GB VRAM
        num_workers=2,
        shuffle=False
    )

    model.to(device)
    model = model.eval()

    with torch.no_grad():
        l2_distance = PairwiseDistance(p=2)
        distances, labels = [], []

        progress_bar = enumerate(tqdm(lfw_dataloader))

        for batch_index, (data_a, data_b, label) in progress_bar:
            data_a = data_a.to(device) # data_a = data_a.cuda()
            data_b = data_b.to(device) # data_b = data_b.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        _, _, _, _, _, _, _, tar, far = evaluate_lfw(distances=distances, labels=labels, far_target=far_target)

        print("TAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(np.mean(tar), np.std(tar), np.mean(far)))


if __name__ == '__main__':
    main()

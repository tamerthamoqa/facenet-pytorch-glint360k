import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str,
                    help="Path to glint360k dataset image folder."
                    )
parser.add_argument('--batch_size', type=int, default=4096,
                    help="Batch Size for iterating through the dataset. (default: 4096)"
                    )
args = parser.parse_args()
dir = args.dir
batch_size = args.batch_size


def calculate_vggface2_rgb_mean_std(dir, batch_size):
    """Calculates the mean and standard deviation of the RGB channels of all images in the glint360k dataset with
       cropped faces by the MTCNN Face Detection model (or any other dataset dataset) when transformed into a Torch
       Tensor of range [0.0, 1.0].

       Method taken from this youtube video by Aladdin Persson: https://www.youtube.com/watch?v=y6IEcEBRZks
    """

    dataset = datasets.ImageFolder(dir, transforms.ToTensor())
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(dataloader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = ((channels_squared_sum / num_batches) - mean ** 2) ** 0.5

    print("Mean: {}, Std: {}".format(mean, std))


if __name__ == '__main__':
    calculate_vggface2_rgb_mean_std(dir=dir, batch_size=batch_size)

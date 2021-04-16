"""Code was inspired by tbmoon's code from his 'facenet' repository
    https://github.com/tbmoon/facenet/blob/master/datasets/write_csv_for_making_dataset.ipynb

    The code was modified to run much faster since 'dataframe.append()' creates a new dataframe per each iteration
    which significantly slows performance.
"""

import os
import glob
import pandas as pd
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generating csv file for triplet loss!")
parser.add_argument('--dataroot', '-d', type=str, required=True,
                    help="(REQUIRED) Absolute path to the dataset folder to generate a csv file containing the paths of the training images for triplet loss training."
                    )
parser.add_argument('--csv_name', type=str,
                    help="Required name of the csv file to be generated. (default: 'glint360k.csv')"
                    )
args = parser.parse_args()
dataroot = args.dataroot
csv_name = args.csv_name


def generate_csv_file(dataroot, csv_name="glint360k.csv"):
    """Generates a csv file containing the image paths of the glint360k dataset for use in triplet selection in
    triplet loss training.

    Args:
        dataroot (str): absolute path to the training dataset.
        csv_name (str): name of the resulting csv file.
    """
    print("\nLoading image paths ...")
    files = glob.glob(dataroot + "/*/*")

    start_time = time.time()
    list_rows = []

    print("Number of files: {}".format(len(files)))
    print("\nGenerating csv file ...")

    progress_bar = enumerate(tqdm(files))

    for file_index, file in progress_bar:

        face_id = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))

        # Better alternative than dataframe.append()
        row = {'id': face_id, 'name': face_label}
        list_rows.append(row)

    dataframe = pd.DataFrame(list_rows)
    dataframe = dataframe.sort_values(by=['name', 'id']).reset_index(drop=True)

    # Encode names as categorical classes
    dataframe['class'] = pd.factorize(dataframe['name'])[0]
    dataframe.to_csv(path_or_buf=csv_name, index=False)

    elapsed_time = time.time()-start_time
    print("\nDone! Elapsed time: {:.2f} minutes.".format(elapsed_time/60))


if __name__ == '__main__':
    generate_csv_file(dataroot=dataroot, csv_name=csv_name)

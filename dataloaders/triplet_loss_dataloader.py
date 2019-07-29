"""This code was imported from tbmoon's 'facenet' repository:
    https://github.com/tbmoon/facenet/blob/master/data_loader.py

    The code was modified to support .png and .jpg files.
"""


import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset


class TripletFaceDataset(Dataset):
    # Modified to add 'training_triplets_path' parameter
    def __init__(self, root_dir, csv_name, num_triplets, training_triplets_path=None, transform=None):

        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name)
        self.num_triplets = num_triplets
        self.transform = transform

        # Modified here
        if training_triplets_path is None:
            self.training_triplets = self.generate_triplets(self.df, self.num_triplets)
        else:
            self.training_triplets = np.load(training_triplets_path)

    @staticmethod
    def generate_triplets(df, num_triplets):

        # Modified here to save the training triplets as a numpy file to not have to redo this process every
        #   training execution from scratch
        def save_triplets(triplets):
            print("Saving training triplets list in datasets/ directory ...")
            np.save('datasets/training_triplets.npy', triplets)
            print("Training triplets' list Saved!\n")

        def make_dictionary_for_face_class(df):

            '''
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            '''
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append(df.iloc[idx, 0])
            return face_classes

        triplets = []
        classes = df['class'].unique()
        face_classes = make_dictionary_for_face_class(df)

        # Modified here to add a print statement
        print("\nGenerating {} triplets...".format(num_triplets))

        progress_bar = tqdm(range(num_triplets))
        for _ in progress_bar:

            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''

            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append(
                [face_classes[pos_class][ianc], face_classes[pos_class][ipos], face_classes[neg_class][ineg],
                 pos_class, neg_class, pos_name, neg_name])

        # Modified here to save the training triplets as a numpy file to not have to redo this process every
        #   training execution from scratch
        save_triplets(triplets=triplets)

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = self.add_extension(os.path.join(self.root_dir, str(pos_name), str(anc_id)))
        pos_img = self.add_extension(os.path.join(self.root_dir, str(pos_name), str(pos_id)))
        neg_img = self.add_extension(os.path.join(self.root_dir, str(neg_name), str(neg_id)))

        # Modified to open as PIL image in the first place
        anc_img = Image.open(anc_img)
        pos_img = Image.open(pos_img)
        neg_img = Image.open(neg_img)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class,
                  'neg_class': neg_class}

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):

        return len(self.training_triplets)

    # Added this method to allow .jpg and .png image support
    def add_extension(self, path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

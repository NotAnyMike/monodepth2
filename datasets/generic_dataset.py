"""
Dataloader for pytorch for dataset with images without intrinsic parameters.
The dataset has to have to files with the path for training images and another
file for validation images.

Author: Mike Woodcock (mike@kopernikusauto.com)
"""

import os
import random
import numpy as np
from pathlib import Path
import torch

from .mono_dataset import MonoDataset

class GenericDataset(MonoDataset):
    """
        Loads the shapes 3dDataset.

        Arguments:
            frame_idxs: the number of the frames. Default: [-1, 0, 1]
            val_samples (str): Str of the path of the txt file with the path for each val sample
            train_samples (str): Str of the path of the txt file with the path for each train sample
    """
    def __init__(self, *args, repeat=1, **kwargs):
        super(GenericDataset, self).__init__(*args, **kwargs)

        self.repeat = int(repeat)

        self.min_frame = abs(min(min(self.frame_idxs), 0))
        self.max_frame = abs(max(max(self.frame_idxs), 0))
        self.filenames_robust = []

        seqs_name = list(set([str(Path(el).parent) for el in self.filenames]))
        seqs = [[el for el in self.filenames if s in el] for s in seqs_name]
        for seq in seqs:
            main = seq[self.min_frame:-self.max_frame]
            post = seq[self.min_frame+self.max_frame:]
            pre = seq[:-self.min_frame-self.max_frame]

            self.filenames_robust.extend(list(zip(main, post, pre)))

    def check_depth(self):
        """Generic datasets do not have depth."""
        return False

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_color(self, folder, do_flip):
        return self.loader(folder)

    def __len__(self):
        # number of elements minus the number of pre and post idx
        return len(self.filenames_robust) * self.repeat

    def __getitem__(self, index):
        """
        Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        index = index % len(self.filenames_robust)
        inputs = {}
        do_flip = self.is_train and random.random() > 0.5

        # Original scale image
        for i, idx in enumerate(self.frame_idxs):
            file_name = self.filenames_robust[index][i]
            inputs[("color", idx, -1)] = self.get_color(file_name, do_flip)

        # TODO color augmentation
        color_aug = (lambda x: x)
        self.preprocess(inputs, color_aug)
        
        for idx in self.frame_idxs:
            del inputs[("color", idx, -1)]
            del inputs[("color_aug", idx, -1)]

        return inputs

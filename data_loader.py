import logging
import pickle
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# All available classes and functions in this module
__all__ = ['MMDataLoader']


# Custom Dataset class for handling multimodal data (text, vision, and audio)
class MMDataset(Dataset):
    def __init__(self, args, mode='test'):
        """
        Initialize the MMDataset.

        :param args: Dictionary containing various configuration settings,
                     including file paths, dataset name, feature dimensions, etc.
        :param mode: Indicates which dataset split to use ('train', 'valid', 'test').
        """
        self.mode = mode
        self.args = args

        # Dataset mapping based on the name, it currently supports only 'Empathy'
        DATASET_MAP = {
            'Empathy': self.__init_evaluation,
        }

        # Initialize the dataset based on the dataset_name in args
        DATASET_MAP[args['dataset_name']]()

    def __init_evaluation(self):
        """
        Initializes the dataset for evaluation purposes by loading the necessary data
        (text, vision, and audio features) and performing some basic preprocessing.
        """
        # Load the dataset from the pickle file
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)

        # Extract the relevant data (text, vision, audio) for the specified mode
        self.text = data[self.mode]['text'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.meta_info = data[self.mode]['info']

        # Labels dictionary to store the labels for the dataset
        self.labels = {'M': np.array(data[self.mode]['labels']).astype(np.float32)}

        # Handle missing (NaN) values by setting them to 0 in each modality
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0
        self.text[self.text != self.text] = 0

        # Update feature dimensions in the args dictionary
        self.args['feature_dims'][0] = self.text.shape[2]
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.args['feature_dims'][2] = self.vision.shape[2]

        # Normalize features if required by args
        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()

    def __normalize(self):
        """
        Normalize the audio and vision features by taking the mean across examples.
        This reduces the variability and removes potential NaN values.
        """
        # Transpose (num_examples, max_len, feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

        # Compute the mean over the sequence length (max_len)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # Handle any remaining NaN values by setting them to 0
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        # Transpose back to the original shape (num_examples, max_len, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        """
        return self.text.shape[0]

    def get_seq_len(self):
        """
        Get the sequence lengths for text, audio, and vision modalities.

        :return: A tuple containing sequence lengths for text, audio, and vision.
        """
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def __getitem__(self, index):
        """
        Get a single sample from the dataset at the specified index.

        :param index: Index of the sample to retrieve.
        :return: A dictionary containing text, audio, vision features, labels, and meta information.
        """
        # Create the sample dictionary with text, audio, vision, and labels
        sample = {
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            "meta_info": self.meta_info[index]  # Meta information related to the sample
        }

        # Add sequence lengths for audio and vision
        sample['audio_lengths'] = self.audio.shape[0]
        sample['vision_lengths'] = self.vision.shape[0]

        return sample


# DataLoader function for handling multiple dataset splits (train, valid, test)
def MMDataLoader(args, num_workers=0):
    """
    Create a DataLoader for each dataset split (train, valid, test) using the MMDataset class.

    :param args: Dictionary containing configuration settings, including file paths, batch size, etc.
    :param num_workers: Number of worker threads to use for loading the data.
    :return: A dictionary of DataLoader objects for 'train', 'valid', and 'test' splits.
    """
    # Create datasets for each split
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    # Get the sequence lengths for each modality if specified in the arguments
    if 'seq_lens' in args:
        args['seq_lens'] = datasets['test'].get_seq_len()

    # Create DataLoader objects for each split
    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=8,  # Default batch size is 8, can be modified
                       num_workers=num_workers,  # Number of worker threads
                       shuffle=False)  # Shuffle is False by default
        for ds in datasets.keys()
    }

    return dataLoader

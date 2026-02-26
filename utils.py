import pandas as pd
from torch.utils.data import Dataset
import pickle


def pickle_variable(variable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)
    print(f"Variable pickled successfully to {filename}")


def unpickle_variable(filename):
    with open(filename, 'rb') as f:
        variable = pickle.load(f)
    print(f"Variable unpickled successfully from {filename}")
    return variable

def import_data():
    training_data = pd.read_csv(
        'https://www.dropbox.com/s/z32q8nks8iqkiuv/waterDataTraining.csv?dl=1',
        index_col=0)
    testing_data = pd.read_csv(
        'https://www.dropbox.com/s/3ptrkyisyks2us3/waterDataTesting.csv?dl=1',
        index_col=0)
    return training_data, testing_data

class WaterQualityDataset(Dataset):
    def __init__(self, features, labels, transform=None, target_transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        value = self.features[idx]
        target = self.labels[idx]
        if self.transform:
            value = self.transform(value)
        if self.target_transform:
            target = self.target_transform(target)
        return value, target
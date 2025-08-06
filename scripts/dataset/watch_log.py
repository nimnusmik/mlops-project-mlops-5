import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from scripts.utils.utils import project_path


class WatchLogDataset:
    def __init__(self, df, scaler=None, label_encoder=None):
        self.df = df
        self.features = None
        self.labels = None
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.contents_id_map = None
        self._preprocessing()

    def _preprocessing(self):
        # label encoding
        if self.label_encoder:
            known_classes = set(self.label_encoder.classes_)
            self.df = self.df[self.df["content_id"].isin(known_classes)].copy()
            self.df["content_id"] = self.label_encoder.transform(self.df["content_id"])
        else:
            self.label_encoder = LabelEncoder()
            self.df["content_id"] = self.label_encoder.fit_transform(self.df["content_id"])

        # mapping
        self.contents_id_map = dict(enumerate(self.label_encoder.classes_))

        # features
        target_columns = ["rating", "popularity", "watch_seconds"]
        features = self.df[target_columns].values

        # scaling
        if self.scaler:
            self.features = self.scaler.transform(features)
        else:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features)

        # one-hot encoding
        num_classes = len(self.label_encoder.classes_)
        encoded_labels = self.df["content_id"].values
        self.labels = np.eye(num_classes)[encoded_labels]

    def decode_content_id(self, encoded_id):
        return self.contents_id_map[encoded_id]

    @property
    def features_dim(self):
        return self.features.shape[1]

    @property
    def num_classes(self):
        return len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def read_dataset(top_k_labels=50):
    path = os.path.join(project_path(), "data", "raw", "watch_log.csv")
    df = pd.read_csv(path)
    top_labels = df["content_id"].value_counts().nlargest(top_k_labels).index
    df = df[df["content_id"].isin(top_labels)].copy()
    return df


def split_dataset(df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
    return train_df, val_df, test_df


def get_datasets(scaler=None, label_encoder=None):
    df = read_dataset()
    train_df, val_df, test_df = split_dataset(df)

    train_dataset = WatchLogDataset(train_df, scaler, label_encoder)

    known_labels = set(train_dataset.label_encoder.classes_)
    val_df = val_df[val_df["content_id"].isin(known_labels)].copy()
    test_df = test_df[test_df["content_id"].isin(known_labels)].copy()

    val_dataset = WatchLogDataset(val_df, scaler=train_dataset.scaler, label_encoder=train_dataset.label_encoder)
    test_dataset = WatchLogDataset(test_df, scaler=train_dataset.scaler, label_encoder=train_dataset.label_encoder)

    return train_dataset, val_dataset, test_dataset
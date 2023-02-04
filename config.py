import torch


class Config:
    def __init__(self):
        self.num_layers = 6
        self.in_points_dim = 3
        self.first_feats_dim = 128
        self.first_subsampling_dl = 0.025
        self.in_feats_dim = 1
        self.conv_radius = 2.5
        self.neighborhood_limits = [32, 32, 32, 32, 16, 16]
        self.deform_radius = 5.

        self.architecture = [
            'conv1d',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'clasify_layer'
        ]


class Config150:
    def __init__(self):
        self.num_layers = 6
        self.in_points_dim = 3
        self.first_feats_dim = 128
        self.first_subsampling_dl = 0.06
        self.in_feats_dim = 1
        self.conv_radius = 2.75
        self.neighborhood_limits = [25, 28, 42, 43, 41, 30]

        self.architecture = [
            'conv1d',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'clasify_layer'
        ]


class ConfigMFR:
    def __init__(self):
        self.num_layers = 6
        self.in_points_dim = 3
        self.first_feats_dim = 128
        self.first_subsampling_dl = 0.025
        self.in_feats_dim = 1
        self.conv_radius = 2.5
        self.neighborhood_limits = [61, 37, 44, 42, 42, 30]
        # self.neighborhood_limits = [20, 20, 20, 20, 20, 20]

        self.architecture = [
            'conv1d',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'scf',
            'random_strided',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'conv1d',
            'nearest_upsample',
            'clasify_layer'
        ]


if __name__ == '__main__':
    import numpy as np
    a = np.array([[1, 2, 3],
                  [3, 3, 1],
                  [1, 4, 7],
                  [1, 5, 6],
                  [2, 6, 8]])
    nx = np.array([[0, 1], [0, 2], [3, 4], [3, 4]], dtype=np.int)
    print(a[nx])
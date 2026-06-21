'''PatchRes anomaly detection module.'''
import torch
from torch import nn
from .ESN_2D_nobatch import ESN_2D
from .common import FaissNN, NearestNeighbourScorer
import numpy as np
import os
from .functions import generate_masks


def window_data_2d(data, window_size, stride):
    """Segments image data into tiles using a 2D sliding window. Used by ESN_extractor."""
    data_list = []
    for j in range(0, data.shape[-2] - window_size[0] + 1, stride):
        for i in range(0, data.shape[-1] - window_size[1] + 1, stride):
            data_list.append(
                data[:, j: j + window_size[0], i: i + window_size[1]])
    return data_list


class ESN_extractor(nn.Module):
    '''Extracting features from image data tiles by computing readout models using a D2-ESN.'''

    def __init__(
        self, input_dim=1, hidden_size=20, window_size=[50, 50], stride=10, spectral_radius=0.9, connectivity=0.1):
        super(ESN_extractor, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.d2esn = ESN_2D(input_dim=input_dim,
                            n_reservoir=hidden_size, alpha=5, spectral_radius=(spectral_radius, spectral_radius), connectivity=connectivity)

    def forward(self, data, device="cpu"):
        '''Tiling and fitting tiles into models.'''
        # patchify
        data_cell = torch.cat(
            window_data_2d(
                data=data, window_size=self.window_size, stride=self.stride)
        )
        data_cell = data_cell.to(device)
        with torch.no_grad():
            features = self.d2esn.forward(data_cell)

        return features
    
    
    def fit_without_tiling(self, data, device="cpu"):
        r'''
        fitting the whole image into a single model
        '''
        data = data.to(device)
        # print("esn data", data.shape)
        features = self.d2esn.forward(data)
        return features


class PatchRes(nn.Module):
    '''PatchRes anomaly detection module.'''

    def __init__(
        self,
        anomaly_score_num_nn=1,
        nn_method=None,
        device="cpu",
        window_size=[50, 50],
        stride=10,
        hidden_size=10,
        graph_channels=1,
        anomaly_threshold=0.5,
        features=''
    ) -> None:
        super(PatchRes, self).__init__()
        self.device = device
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        self.window_size = window_size
        self.stride = stride
        self.hidden_size = hidden_size

        self.extractor = ESN_extractor(
            input_dim=graph_channels,
            window_size=self.window_size,
            hidden_size=self.hidden_size,
            stride=self.stride
        )
        self.anomaly_threshold = anomaly_threshold
        if os.path.isfile(features):
            self.features = torch.load(features)
        else:
            self.features = None

    def fit(self, images):
        if self.features is None:
            r'''fill model bank with models of training set'''
            features = self.extractor(images, device=self.device)
        else:
            features = self.features
        self.anomaly_scorer.fit(detection_features=features)
        return features

    def predict(self, images, mode="OD", return_box=False, normalization=True):
        """
        Scores tiles according to k-neighbor distance to the normal model bank,
        and reshapes anomaly scores of tiles to perform pixel segmentation or frame out the anomaly region.

        Parameters:
        -----------
        images : torch.Tensor
            Input images with shape [batch_size, H, W].
            
        mode : str, optional
            Mode of operation, either "pixel_seg" for pixel segmentation or "OD" for framing out the anomaly region (default is "OD").
        
        return_box : bool, optional
            Whether to return bounding boxes for anomalies (default is False).
        
        normalization : bool, optional
            Whether to normalize scores (default is True).

        Returns:
        --------
        tuple
            Returns a tuple containing:
            - masks : torch.Tensor
                Segmentation masks for the images. Mask type determined by mode.
            - patch_masks : torch.Tensor
                Tile scores.
            - features : numpy.ndarray
                Extracted features from the images.
            - patch_scores : numpy.ndarray
                Anomaly scores for the image tiles.
            - extracted_image : torch.Tensor
                Extracted anomaly region.
        """
        images = images.to(torch.float).to(self.device)
        features = self.extractor(images)
        with torch.no_grad():
            # scoring patches
            features = np.asarray(features)
            tile_scores = self.anomaly_scorer.predict(features)[0]

            if normalization:
                tile_scores = (tile_scores - tile_scores.min()) / (
                    tile_scores.max() - tile_scores.min()
                )

            patch_masks, masks, extracted_image = generate_masks(
                images, mode=mode, window_size=self.window_size, anomaly_threshold=self.anomaly_threshold, tile_scores=tile_scores, stride=self.stride, normalization=normalization, return_box=return_box)
            return masks, patch_masks, features, tile_scores, extracted_image

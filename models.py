"""Style transfer model

This module contains the neural network that is used for neural style transfer as specified in this paper:
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import utils

from torchvision import transforms, models

class StyleTransferModel(nn.Module):
    """Class of the neural transfer network used for feature extraction
    ...

    Attributes
    ----------
    model : torch.nn.Sequential
        pretrained convolutional layers of the VGG19 network based
    
    Methods
    -------
    forward(x)
        Passes data x through the network
    get_specified_feature_maps(x, selected_layers)
        Passes data x through the network and returns specified feature maps
    """

    def __init__(self):
        super(StyleTransferModel, self).__init__()
        self.model = models.vgg19(pretrained=True).features.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        """Passes data x through the network
        
        Parameters
        ----------
        x : torch.Tensor
            Input data that is forward propagated

        Returns
        -------
        torch.Tensor
            Output of network
        """

        return self.model(x)

    def get_specified_feature_maps(self, x, selected_layers=None):
        """Passes data x through the network and returns specified feature maps
        This function is based on Udacity's implementation found here:
        https://github.com/udacity/deep-learning-v2-pytorch/blob/master/style-transfer/Style_Transfer_Solution.ipynb
        
        Parameters
        ----------
        x : torch.Tensor
            Input data that is forward propagated
        selected_layers : dict
            Dictionary where the keys are the layer indices

        Returns
        -------
        dict <torch.Tensor>
            Selected feature maps
        """
        if selected_layers is None:
            selected_layers={'0':'conv1_1',
                             '5':'conv2_1',
                             '10':'conv3_1',
                             '19':'conv4_1',
                             '21':'conv4_2', # This is the content layer as specified in the paper
                             '28':'conv5_1'}

        selected_feature_maps = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in selected_layers:
                selected_feature_maps[selected_layers[name]] = x

        return selected_feature_maps

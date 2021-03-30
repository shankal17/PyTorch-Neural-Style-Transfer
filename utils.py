"""This module contains useful functions for the neural style transfer task"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image
from matplotlib import rcParams, cm

def convert_tensor_to_numpy_img(tensor_img):
    """Converts image of type torch.tensor to numpy.ndarray for matplotlib display image

    Parameters
    ----------
    tensor_img : torch.Tensor
        Input image to be converted to

    Returns
    -------
    numpy.ndarray
        Numpy representation of the input image
    """

    img = tensor_img.to('cpu').clone().detach()
    img = img.numpy().squeeze(0)
    img = img.transpose(1, 2, 0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    img = img.clip(0, 1)
    return img

def display_side_by_side_imgs(left_img, right_img, left_title='Left Image', right_title='Right Image', figsize=(16, 8)):
    """Displays two images side by side

    Parameters
    ----------
    left_img : torch.Tensor or numpy.ndarray
        Image that is to be displayed on the left
    right_img : torch.Tensor or numpy.ndarray
        Image that is to be displayed on the right
    left_title : string
        Title of the left subplot
    right_title : string
        Title of the right subplot
    figsize : tuple
        Size of the matplotlib figure
    """

    # Convert the images to numpy arrays if they aren't already
    if isinstance(left_img, torch.Tensor):
        left_img = convert_tensor_to_numpy_img(left_img)
    if isinstance(right_img, torch.Tensor):
        right_img = convert_tensor_to_numpy_img(right_img)

    # Create plots and set titles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(left_img)
    ax2.imshow(right_img)
    ax1.title.set_text(left_title)
    ax2.title.set_text(right_title)
    plt.show()

def calculate_gram_matrix(x):
    """Calculates the gram matrix of the input tensor

    Parameters
    ----------
    x : torch.Tensor
        Input tensor

    Returns
    -------
    torch.Tensor
        The calculated gram matrix of the input tensor
    """

    batch_size, depth, height, width = x.size()
    x = x.view(depth, height*width)
    gram_matrix = torch.mm(x, x.t())
    return gram_matrix

def load_content_and_style_images(content_dir, style_dir, max_allowable_size=400, resize_shape=None):
    """Loads the content and style images and resizes them if appropriate

    Parameters
    ----------
    content_dir : string
        File location of the content image
    style_dir : string
        File location of the style image
    max_allowable_size : int
        The maximum size of the images to prevent slow processing
    resize_shape : int or sequence, optional
        Shape in which to reshape the images

    Returns
    -------
    torch.Tensor
        These are the loaded content and style images
    """

    content_img = Image.open(content_dir).convert('RGB')
    style_img = Image.open(style_dir).convert('RGB')

    if max(content_img.size) > max_allowable_size:
        size = max_allowable_size
    else:
        size = max(content_img.size)

    if resize_shape is not None:
        size = resize_shape

    content_img_transforms = transforms.Compose([transforms.Resize(size),\
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), 
                                                                      (0.229, 0.224, 0.225))])

    content_img = content_img_transforms(content_img)[:3,:,:].unsqueeze(0)

    style_img_transforms = transforms.Compose([transforms.Resize(content_img.shape[-2:]),\
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406), 
                                                                    (0.229, 0.224, 0.225))])

    style_img = style_img_transforms(style_img)[:3,:,:].unsqueeze(0)

    return content_img, style_img

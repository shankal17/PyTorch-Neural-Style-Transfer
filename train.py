"""This module contains a function that encompasses the training process"""

import models
import utils
import torch.optim as optim
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

def train(content_img, style_img, epochs=5000, intitial_lr=0.003, style_layer_weights=None, content_weight=1, style_weight=1e6, print_every=20, save_name=None):
    """Conducts neural style transferred

    Parameters
    ----------
    content_img : torch.Tensor
        Image to be used as the content information
    style_img : torch.Tensor
        Image to be used as for style information
    epochs : int
        Number of iterations to produce the final image
    initial_lr : float, optional
        Initial learning rate used by the optimizer
    style_layer_weights : dict, optional
        Weights to be used on each feature map used to extract style information
    content_weight : float, optional
        Weight associated with the content loss portion of the total loss
    style_weight : float, optional
        Weight associalted with the style loss portion of the total loss
    print_every : int, optional
        Interval to display intermediate results to
    save_name : string, optional
        Name to save the final image as

    Returns
    -------
    torch.Tensor
        The final image that retains the content of the content image with the style of the style image
    """

    # Load model used for feature extraction
    cnn = models.StyleTransferModel()

    # Move everything to available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img, style_img = content_img.to(device), style_img.to(device)
    cnn.model.to(device)

    # Initalize the final image as the content image since that is relatively close to the final result
    target = content_img.clone().requires_grad_(True)

    # Generate the feature maps of the content image and style image
    content_img_features = cnn.get_specified_feature_maps(content_img)
    style_img_features = cnn.get_specified_feature_maps(style_img)

    # Calculate the gram matrices of the style image feature maps and define default style layer weights
    # This portion is from here: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/style-transfer/Style_Transfer_Solution.ipynb
    # It was the best and most flexible implementation
    style_gram_matrices = {layer: utils.calculate_gram_matrix(style_img_features[layer]) for layer in style_img_features}
    if style_layer_weights is None:
        style_layer_weights = {'conv1_1':1.0,
                               'conv2_1':0.5,
                               'conv3_1':0.5,
                               'conv4_1':0.5,
                               'conv5_1':0.5}

    # Create optimizer. Note that it operates on the target image
    optimizer = optim.Adam([target], lr=intitial_lr)

    # Run through each epoch
    for i in range(epochs):
        # Generate feature maps of the current target image
        target_img_features = cnn.get_specified_feature_maps(target)

        # Calculate content loss
        content_loss = F.mse_loss(content_img_features['conv4_2'], target_img_features['conv4_2'])

        # Calculate style loss
        style_loss = 0
        for layer_name in style_layer_weights:
            target_feature = target_img_features[layer_name]
            target_gram = utils.calculate_gram_matrix(target_feature)
            style_gram = style_gram_matrices[layer_name]
            layer_style_loss = style_layer_weights[layer_name]*F.mse_loss(style_gram, target_gram)
            style_loss += layer_style_loss

        # Calculate total aggregate loss
        total_loss = content_weight*content_loss + style_weight*style_loss

        # Update target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
        # Print loss and display image on specified interval
        if i % print_every == 0:
            print("epoch {}: Total Loss: {}".format(i, total_loss.item()))
            plt.imshow(utils.convert_tensor_to_numpy_img(target))
            plt.show()

    if save_name is not None:
        plt.imshow(utils.convert_tensor_to_numpy_img(target))
        plt.savefig(save_name)

    return target

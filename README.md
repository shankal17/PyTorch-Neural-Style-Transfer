# PyTorch-Neural-Style-Trasfer

This project is based on the [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) paper.

![Eye Fractal](/results/eye_fractal/eye_content_fractal_style.png)

## Getting Started

Once you have the code, set up a virtual environment if you would like and install the necessary libraries by running the command below.
```bat
pip install -r /path/to/requirements.txt
```

From there, just open [Example.ipynb](https://github.com/shankal17/PyTorch-Neural-Style-Transfer/blob/main/Example.ipynb) and use your own content and style images! You can also play with the [style layer weights](https://github.com/shankal17/PyTorch-Neural-Style-Transfer/blob/main/train.py#:~:text=style_layer_weights%20%3D%20%7B%27conv1_1%27%3A1.0,%27conv5_1%27%3A0.5%7D) if you would like.

## Some of my Results

![Bridge Tree](/results/bridge_tree/Bridge_Tree_Result.JPG)

![Scissor Dance](/results/scissor_dance/scissor_dance_result.JPG)

![Eye Fractal](/results/eye_fractal/eye_fractal_result.JPG)

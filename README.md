# Style Transfer Project

### Team members

| Name | Email | Group |
| --- | --- | --- |
| Andery Vagin | a.vagin@innopolis.university | B20-AI-01 |
| Nguyen Gia Trong | g.nguyen@innopolis.university | B20-AI-01 |
| Sergey Golubev | s.golubev@innopolis.university | B20-AI-01 |

## Project Topic

To build **an image stylization model** using generative AI techniques. The project is aimed at implementing three different models to apply various artistic styles to input images.

![image](https://github.com/KKroliKK/style-transfer-project/assets/48735488/5a24664a-3646-49cd-888d-40a0e72daf32)

## Key Objectives:

- **Research on style transfer approaches**: analyze common techniques and SOTA solutions.
- **Select models to implement**: decide on the types of models. For today, we are going to research on the following architectures applied to image stylization:
    - convolutional neural networks
    - generative adversarial networks
    - transofrmer architectures
- **Choose datasets**: acquire several datasets, decide whether to merge them or not for models training.
- **Train the selected models** on the datasets, apply fine-tuning.
- **Comparative analysis**: what is common and different in the implemented models? How different are the results when the same input images are provided?
- **Deployment:** we are going to implement our solution as a Telegram bot. The project could be used as an art and design helper, and a Telegram bot is a suitable deployment format.

# Progress So Far

## 1. ****Neural Algorithm of Artistic Style****

We have implemented a style transfer model based on the [Neural-Style algorithm by Gatys et al](https://arxiv.org/abs/1508.06576). The key principle behind this model is described below.

We have two input images (input-content and input-style) and one output image. We introduce two distances, **a content distance Dc and a style distance Ds.**

- **Dc** measures the content difference between input-content and output.
- **Ds** measures the style difference between input-style and output.

The model transforms an input image by **minimizing Dc and Ds at the same time**. Style transfer here is performed via increasing the pairwise correlations between output and input feature maps.

The pipeline is as follows:

1. A user uploads input-content and input-style images. The images get resized and normalized.
2. **Loss Functions**:
    - Dc represents **Content Loss**. It computes the mean squared error (MSE) between the feature map of the layer in the input and content images.
    - Ds represents **Style Loss**: It measures the style difference between two images. It computes the Gram matrix for the input-style and output feature maps.
        
        ![image](https://github.com/KKroliKK/style-transfer-project/assets/48735488/86e0e93d-35db-43ed-b4aa-877017e9e235)
        
3. We use **pre-trained VGG19 convolutional neural network as feature extractor**. Its implementation from Pytorch is trained on ImageNet dataset. We also normalize inputs to preserve the same distributions observed during training the VGG19.
4. The style transfer model is essentially the VGG feature extractor with the two losses computed at the specified layers.
5. We use **L-BFGS optimizer to run gradient descent**. In the considered approach, we don’t want to train a network, but we want to train the input image by minimize the content and style losses. Images have h*w*c parameters which is often a big number, and L-BFGS optimization algorithm handles such tasks well.
6. **Inference**: the input image is iteratively passed through the network, becoming a more and more stylised version of itself.

- You may find the **inference examples** in `data` folder of this repo.

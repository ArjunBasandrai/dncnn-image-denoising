<h1 align="center">DnCNN Image Denoising</h1>

<p align="center">
  PyTorch implementation of <a href="https://arxiv.org/pdf/1608.03981">"Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"</a> trained on the <a href="[https://www.kaggle.com/datasets/wenewone/cub2002011](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500)">Berkeley Segmentation Dataset 500 (BSDS500)</a> dataset with synthetically added Gaussian Noise of varying noise level
</p>

# Model Architecture

The model is built using the DnCNN architecture described in paper with network depth of 21

![image](https://github.com/user-attachments/assets/5a320510-a2dd-4a37-b72c-a70970d01430)

## Training Data

The model is trained on the <a href="[https://www.kaggle.com/datasets/wenewone/cub2002011](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500)">Berkeley Segmentation Dataset 500 (BSDS500)</a> datatset available on Kaggle

## Training Results

**MSE**: Mean Squared Error

| Network Depth | Training Loss (MSE) | Validation Loss (MSE) |
| :---: | :-----------: | :-------------: |
| 21 | 0.0002 | 0.0002 |

## Denoising Results

Following are some of the results of the model on a few images with added gaussian noise from the test and validation sets of the dataset

![image](https://github.com/user-attachments/assets/38dd5c63-78f5-4340-84d3-d0461312d12d)
![image](https://github.com/user-attachments/assets/af5e63bf-e04a-4c36-a978-afa8d13f9926)
![image](https://github.com/user-attachments/assets/59ff8820-4370-4604-904b-69e126ef128c)

## Usage

1. Install dependencies
```bash
pip install torch torchvision matplotlib pillow numpy
```
2. Download the inference script located at `dncnn-image-denoising-inference.ipynb`
3. Update the model path in the inference script, adjust image links as needed, and execute the cells.

## References

1. Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2016, August 13). Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. arXiv. https://arxiv.org/pdf/1608.03981

2. Martin, D., Fowlkes, C., Tal, D., & Malik, J. (2001, July). A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In Proceedings of the 8th International Conference on Computer Vision (Vol. 2, pp. 416–423).

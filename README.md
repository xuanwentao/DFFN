# A New Dual-Feature Fusion Network for Enhanced Hyperspectral Unmixing
Xuanwen Tao, Bikram Koirala, Antonio Plaza, Paul Scheunders

**1. Abstract**

Hyperspectral unmixing is a crucial technique in remote sensing data processing that aims to estimate component information from mixed pixels in hyperspectral images. Most existing deep learning-based hyperspectral unmixing models employ autoencoder networks to reconstruct hyperspectral images and estimate abundance maps. Here, the weight between the reconstructed and the Softmax layers is used to extract/estimate endmember signatures. However, autoencoders are heavily dependent on initial weights, which introduces inherent randomness, potentially compromising unmixing accuracy. To address this issue, in this paper we present a new dual-feature fusion network (DFFN) for enhanced hyperspectral unmixing. Our DFFN mainly consists of four modules: 1) a feature fusion module (FFM), 2) an abundance estimation module (AEM), 3) an endmember estimation module (EEM), and 4) a reconstruction module (RE). Firstly, FFM calculates spectral and spatial similarities and then enhances the hyperspectral image by matrix multiplications with similarity matrices. Second, AEM takes the enhanced hyperspectral image as input and uses convolutional layers to estimate abundances and reconstruct the image. Next, the reconstructed image  is fed into EEM to automatically estimate endmembers. RE performs the final reconstruction through matrix multiplication of the estimated endmembers and abundances.

**2. Overview**

![2-A2SAN](https://github.com/xuanwentao/DFFN/blob/main/DFFN1.png)


**3. Citation**

Please kindly cite the paper if this code is useful and helpful for your research.

X. Tao, B. Koirala, A. Plaza and P. Scheunders, "A New Dual-Feature Fusion Network for Enhanced Hyperspectral Unmixing," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-13, 2024, Art no. 5540113, doi: 10.1109/TGRS.2024.3505292.

     @article{tao2024new,
      title={A New Dual-Feature Fusion Network for Enhanced Hyperspectral Unmixing},
      author={Tao, Xuanwen and Koirala, Bikram and Plaza, Antonio and Scheunders, Paul},
      journal={IEEE Transactions on Geoscience and Remote Sensing},
      year={2024},
      volume={62},
      number={},
      pages={1-13},
      publisher={IEEE}
      }

**4. Contact Information**

Xuanwen Tao: txw_upc@126.com<br> 

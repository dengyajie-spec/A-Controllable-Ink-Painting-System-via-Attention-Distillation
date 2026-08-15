# A Controllable Ink Painting System via Attention Distillation
This project builds an interactive ink painting generation system based on diffusion models, aiming to lower the barrier to creating traditional Chinese ink art. The system is built around Stable Diffusion 1.5 and integrates three core modules: fine‑tuning the VAE decoder, attention‑distillation‑based style transfer, and an image refinement component leveraging ControlNet with Canny edge constraints. Users only need to upload a content image and a style reference, and can then adjust the generation results in real time using four sliders—style preservation, ink density, negative space, and brush wetness—to achieve personalised aesthetic control. The system is packaged as a lightweight Flask web application with support for both Chinese and English interfaces.

## 1. Preparations

If you want to retrain the model, you can follow the steps below to replicate it.

### 1.1 Environment and Usage Instructions​

- Linux
- Python 3.10
- CUDA 12.4
   ```
   pip install torch transformers datasets peft bitsandbytes flask
   ```
  
### 1.2 Project Preparation

   (1) Stable Diffusion 1.5：[Download from Hugging Face](https://huggingface.co/Jiali/stable-diffusion-1.5/tree/main)

   (2) ControlNet：[Download from Hugging Face](https://huggingface.co/lllyasviel/sd-controlnet-canny/tree/main)

   (3) Project Structure：<details><summary>📁（Click to expand）  </summary>

```bash
├── stable-diffusion-v1-5/
├── controlnet-canny/
├── static/
│   ├── css/style.css
│   └── js/script.js
├── templates/index.html
├── app.py
├── losses.py
├── pipeline_sd.py
├── vae_trainer.py
├── utils.py
├── 1.VAE Decoder Fine-Tuning.ipynb
├── 2.Attention Distillation.ipynb
└── 3.ControlNet with Canny Edge Detection.ipynb
```
 </details>

## 2. Jupyer notebook
This repository contains three Jupyter Notebooks covering the project's key procedures, enabling users to perform detection in separate steps.Note that running these notebooks requires supporting external files, so please ensure all file paths are correctly configured.

## 3. Quick Start

You can experience this project directly without training the model by following these steps:

   (1) Place the files in the correct paths
   
   (2) Open the folder in the terminal

   (3) Run the code
   ```
   python app.py
   ```
   (4) Open the link in the terminal

   (5) ​​Visual Results ​：

   <img width="1024" height="1053" alt="39XQ4SG{H5D0JW$5QL5 YMN" src="https://github.com/user-attachments/assets/ed0bb5d5-0ef0-4ebf-b7a2-506a69c018d1" />


   (5) ​​Comparison of Different Effects ：

   <img width="370" height="501" alt="image" src="https://github.com/user-attachments/assets/d6065689-ab04-4e43-b95b-ad54bab4f7c1" />


   # References
Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6), 679-698.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021, October). Emerging properties in self-supervised vision transformers. In 2021 IEEE/CVF international conference on computer vision (ICCV) (pp. 9630-9640). IEEE.

Chen, X., Xu, C., Yang, X., Song, L., & Tao, D. (2018). Gated-gan: Adversarial gated networks for multi-collection style transfer. IEEE Transactions on Image Processing, 28(2), 546-560.
Choi, Y., Choi, M., Kim, M., Ha, J. W., Kim, S., & Choo, J. (2018, June). Stargan: Unified generative adversarial networks for multi-domain image-to-image translation. In 2018 IEEE/CVF conference on computer vision and pattern recognition (pp. 8789-8797). IEEE.

Chung, J., Hyun, S., & Heo, J. P. (2024, June). Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 8795-8805). IEEE.

Cross, G. R., & Jain, A. K. (1983). Markov random field texture models. IEEE Transactions on pattern analysis and machine intelligence, (1), 25-39.

Efros, A. A., & Freeman, W. T. (2023). Image quilting for texture synthesis and transfer. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2 (pp. 571-576).

Efros, A. A., & Leung, T. K. (1999, September). Texture synthesis by non-parametric sampling. In Proceedings of the seventh IEEE international conference on computer vision (Vol. 2, pp. 1033-1038). IEEE.

Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576.

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. Advances in neural information processing systems, 27.

He, B., Gao, F., Ma, D., Shi, B., & Duan, L. Y. (2018, October). Chipgan: A generative adversarial network for chinese ink wash painting style transfer. In Proceedings of the 26th ACM international conference on Multimedia (pp. 1172-1180).

He, X., Zhu, M., Wang, N., Wang, X., & Gao, X. (2023). BiTGAN: bilateral generative adversarial networks for Chinese ink wash painting style transfer. Science China Information Sciences, 66(1), 1-2.

Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598.

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems, 33, 6840-6851.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). Lora: Low-rank adaptation of large language models. Iclr, 1(2), 3.

Huang, X., & Belongie, S. (2017, October). Arbitrary style transfer in real-time with adaptive instance normalization. In 2017 IEEE international conference on computer vision (ICCV) (pp. 1510-1519). IEEE.

Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Liu, H., Song, Z., Wang, Y., Hu, B., & Wang, Y. (2026). ChipDiff: Staged diffusion model with loss gradient guidance for Chinese ink painting style transfer. Pattern Recognition, 113309.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022, June). High-resolution image synthesis with latent diffusion models. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR) (pp. 10674-10685). ieee.

Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015, June). Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning (pp. 2256-2265). pmlr.

Wang, Z., Zhao, L., & Xing, W. (2023, October). Stylediffusion: Controllable disentangled style transfer via diffusion models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 7643-7655). IEEE.

Xie, S., & Tu, Z. (2015). Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision (pp. 1395-1403).

Xue, A. (2021, January). End-to-end chinese landscape painting creation using generative adversarial networks. In 2021 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 3862-3870). IEEE.

Zhang, L., Rao, A., & Agrawala, M. (2023, October). Adding conditional control to text-to-image diffusion models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 3813-3824). IEEE.

Zhang, Y., Huang, N., Tang, F., Huang, H., Ma, C., Dong, W., & Xu, C. (2023, June). Inversion-based style transfer with diffusion models. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 10146-10156). IEEE.

Zhou, Y., Gao, X., Chen, Z., & Huang, H. (2025, June). Attention distillation: A unified approach to visual characteristics transfer. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 18270-18280). IEEE.

Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017, October). Unpaired image-to-image translation using cycle-consistent adversarial networks. In 2017 IEEE international conference on computer vision (ICCV) (pp. 2242-2251). Ieee.

Zhu, J. Y., Zhang, R., Pathak, D., Darrell, T., Efros, A. A., Wang, O., & Shechtman, E. (2017). Toward multimodal image-to-image translation. Advances in neural information processing systems, 30.


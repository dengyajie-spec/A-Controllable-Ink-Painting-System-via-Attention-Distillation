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

   (1) Stable Diffusion 1.5：[Download from Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5)

   (2) ControlNet：[Download from Hugging Face](https://huggingface.co/lllyasviel/ControlNet)

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

   <img width="431" height="391" alt="image" src="https://github.com/user-attachments/assets/ca9e2368-dc8d-44a8-b4ff-b5c7f9504d83" />


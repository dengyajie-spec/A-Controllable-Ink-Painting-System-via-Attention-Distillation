# A Controllable Ink Painting System via Attention Distillation
This project builds an interactive ink painting generation system based on diffusion models, aiming to lower the barrier to creating traditional Chinese ink art. The system is built around Stable Diffusion 1.5 and integrates three core modules: fine‑tuning the VAE decoder, attention‑distillation‑based style transfer, and an image refinement component leveraging ControlNet with Canny edge constraints. Users only need to upload a content image and a style reference, and can then adjust the generation results in real time using four sliders—style preservation, ink density, negative space, and brush wetness—to achieve personalised aesthetic control. The system is packaged as a lightweight Flask web application with support for both Chinese and English interfaces.

## 1. Preparations

If you want to retrain the model, you can follow the steps below to replicate it.

### 1.1 Environment and Usage Instructions​

- Linux
- Python 3.10
- CUDA 12.1
   ```
   pip install torch transformers datasets peft bitsandbytes flask
   ```
  
### 1.2 Project Preparation

   (1) Stable Diffusion 1.5：[Download from Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5)

   (2) ControlNet：[Download from Hugging Face](https://huggingface.co/lllyasviel/ControlNet)

   (3) Project Structure：<details><summary>📁（Click to expand）  </summary>

```bash
├── deepseek/
├── lora_honglou/
├── lora_sanguo/
├── lora_shuihu/
├── lora_xiyou/
├── static/
│   ├── css/style.css
│   └── js/script.js
├── templates/index.html
├── app.py
├── Test.ipynb
└── Train.ipynb
```
 </details>

## 2. Jupyer notebook
This repository contains three Jupyter Notebooks covering the project's key procedures, enabling users to perform detection in separate steps.Note that running these notebooks requires supporting external files, so please ensure all file paths are correctly configured.
### 2.1 Model Training​ 

   (1) Dataset：This project uses the open-source [StyleLLM](https://github.com/stylellm/stylellm_models) dataset, which contains excerpts from the Four Great Classical Novels of Chinese literature.

   (2) Model Training：Open Train.ipynb, make sure to modify the dataset path, then run the code to start model training.

   <img width="500" height="200" alt="%EU_N`R65IXZ~A~2(NVI()I" src="https://github.com/user-attachments/assets/285182ac-0f69-4fdd-9e3f-4ea805be61d7" />


   (3) Model Testing：Open Test.ipynb. You can enter text as needed to check whether the model is running properly.
   
   <img width="500" height="100" alt="7384176WJ 0K5XH@U 0 7C" src="https://github.com/user-attachments/assets/853f1109-6640-4d63-919d-fb8d6ee50e11" />

   Note：You can adjust the maximum number of output characters by modifying the value of MAX_NEW_TOKENS.
  
## 3 Quick Start

You can experience this project directly without training the model by following these steps:

   (1) Place the files in the correct paths
   
   (2) Open the folder in the terminal

   (3) Run the code
   ```
   python app.py
   ```
   (4) Open the link in the terminal

   (5) ​​Visual Results ​：

   <img width="305" height="263" alt="image" src="https://github.com/user-attachments/assets/9e80cc3a-dec9-4a82-98df-a95f95144775" />

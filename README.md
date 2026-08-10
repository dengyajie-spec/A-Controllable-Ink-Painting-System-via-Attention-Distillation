# A Controllable Ink Painting System via Attention Distillation
This is a lightweight text style transfer system based on the DeepSeek large language model, built using LoRA (Low-Rank Adaptation), a parameter-efficient fine-tuning technique. This project focuses on converting modern Chinese text into the classical literary styles of China’s Four Great Classical Novels: Dream of the Red Chamber, Romance of the Three Kingdoms, Water Margin, and Journey to the West. It achieves high-quality style transfer while preserving the original semantic meaning. In addition, the project supports hybrid style generation, which fuses two different styles.

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

   (1) DeepSeek：[Download from Hugging Face](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base/tree/main)

   (2) Project Structure：<details><summary>📁（Click to expand）  </summary>

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

## 2. Model
If you want to train your own model with your private dataset, please follow the steps below. If you just want to run this project directly, please refer to the third part directly.
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

---
title: Lora Cerebras Gpt2.7b Alpaca Shortprompt
emoji: 🐨
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 3.23.0
app_file: app.py
pinned: false
license: apache-2.0
---

### 🦙🐕🧠 Cerebras-GPT2.7B LoRA Alpaca ShortPrompt

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lxe/cerebras-lora-alpaca/colab.ipynb)
[![Open In Spaces](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/lxe/lxe-Cerebras-GPT-2.7B-Alpaca-SP)
[![](https://img.shields.io/badge/no-bugs-brightgreen.svg)](https://github.com/lxe/no-bugs) 
[![](https://img.shields.io/badge/coverage-%F0%9F%92%AF-green.svg)](https://github.com/lxe/onehundred/tree/master)

Scripts to finetune [Cerebras GPT2.7B](https://www.cerebras.net/blog/cerebras-gpt-a-family-of-open-compute-efficient-large-language-models/) on the [Alpaca dataset](https://github.com/gururise/AlpacaDataCleaned), as well as inference demos. 

 - It is the fastest model in the west!
 - The model with LoRA weights merged-in available at [HuggingFace/lxe/Cerebras-GPT-2.7B-Alpaca-SP](https://huggingface.co/lxe/Cerebras-GPT-2.7B-Alpaca-SP)
 - The LoRA weights also available at [HuggingFace/lxe/lora-cerebras-gpt2.7b-alpaca-shortprompt](https://huggingface.co/lxe/lora-cerebras-gpt2.7b-alpaca-shortprompt)
 - [ggml](https://github.com/ggerganov/ggml) version of the model available at [HuggingFace/lxe/ggml-cerebras-gpt2.7b-alpaca-shortprompt](https://huggingface.co/lxe/Cerebras-GPT-2.7B-Alpaca-SP-ggml). You can run this without a GPU and it's much faster than the original model

<img src="https://user-images.githubusercontent.com/1486609/229048081-57629025-cf4e-4771-9872-f10ee90751b1.gif" width="400" />

### 📈 Warnings

The model tends to be pretty coherent, but it also hallucinates a lot of factually incorrect responses. Avoid using it for anything requiring factual correctness.

### 📚 Instructions

0. Be on a machine with an NVIDIA card with 12-24 GB of VRAM.

1. Get the environment ready

```bash
conda create -n cerebras-lora python=3.10
conda activate cerebras-lora
conda install -y cuda -c nvidia/label/cuda-11.7.0
conda install -y pytorch=1.13.1 pytorch-cuda=11.7 -c pytorch
```

2. Clone the repo and install requirements

```
git clone https://github.com/lxe/cerebras-lora-alpaca.git && cd !!
pip install -r requirements.txt
```

3. Run the inference demo

```
python app.py
```

To reproduce the finetuning results, do the following:

3. Install jupyter and run it

```
pip install jupyter
jupyter notebook
```

4. Navigate to the `inference.ipynb` notebook and test out the inference demo.

5. Navigate to the `finetune.ipynb` notebook and reproduce the finetuning results.

 - It takes about 5 hours with the default settings
 - Adjust the batch size and gradient accumulation steps to fit your GPU

### 📝 License

Apache 2.0

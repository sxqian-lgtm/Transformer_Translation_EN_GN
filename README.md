# Transformer_Translation_EN_GN

A fully manual, from-scratch implementation of the Transformer architecture for English → German translation. This project includes clean, modular implementations of all major Transformer components, along with Jupyter notebooks for step‑by‑step explanation and visualization.

---

## 🚀 Project Overview

This repository contains a complete hand-written Transformer model built using PyTorch, without relying on `nn.Transformer`. Every module is implemented manually, including:

- Multi‑Head Attention  
- Positional Encoding  
- Feed‑Forward Network  
- Encoder & Decoder Layers  
- Full Encoder‑Decoder Architecture  
- Forward & Backward Pass  
- Data Processing Pipeline  

The goal is to deeply understand how the Transformer works internally by building each part from scratch.

---

## 📂 Repository Structure


Transformer_Translation_EN_GN/ │ ├── Encoder_Decoder_Layers.ipynb ├── Encoder_Decoder_Layers.py ├── Feed_Forward_Network.ipynb ├── Feed_Forward_Network.py ├── Forward_Backward_Pass.ipynb ├── Forward_Backward_Pass.py ├── Multi_Head_Attention.ipynb ├── Multi_Head_Attention.py ├── Position_Encoding.ipynb ├── Position_Encoding.py ├── Sample_Dataset.ipynb ├── Sample_Dataset.py ├── Text_Processing.ipynb ├── Text_Processing.py ├── Transformer.ipynb ├── Utils.ipynb ├── Utils.py │ ├── .gitignore └── README.md

Each `.ipynb` notebook explains the logic step‑by‑step. Each `.py` file contains the clean, reusable implementation.

---

## 🧠 Key Features

- Manual implementation of **Scaled Dot‑Product Attention**
- Manual implementation of **Multi‑Head Attention**
- Custom **Positional Encoding** (sinusoidal)
- Layer‑normalized **Encoder & Decoder blocks**
- Custom **masking logic** (padding mask + causal mask)
- Full **training forward/backward pass**
- Clean modular code for easy debugging and visualization

---

## 🛠️ Requirements


Python 3.10+ PyTorch numpy tqdm

---

## 📘 How to Use

### 1. Explore the notebooks  
Start with:

- `Position_Encoding.ipynb`
- `Multi_Head_Attention.ipynb`
- `Encoder_Decoder_Layers.ipynb`
- `Transformer.ipynb`

Each notebook builds on the previous one.

### 2. Run the full model  
Use `Transformer.ipynb` to run the full forward pass and test translation.

---

## 📌 Notes

- Large datasets (Multi30k, OPUS100) and model checkpoints are **excluded** from the repository to comply with GitHub’s file size limits.
- You can download datasets separately or use your own.

---

## ✨ Acknowledgements

Inspired by the original Transformer paper:  
**“Attention Is All You Need” (Vaswani et al., 2017)**  
and by the goal of deeply understanding the architecture through manual implementation.




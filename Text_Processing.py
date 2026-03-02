#!/usr/bin/env python
# coding: utf-8

# ## Description
# This notebook will transform the raw datasets into a format suitable for training and evaluation. It will open datasets by the file: Simple_Dataset.ipynb. What's more, this project will transform the data to vector format, which is the input and output of the model, including tokenization, and creating vocabulary for the Transformer model.
# 

# In[4]:


## Introducing the data
from Sample_Dataset import load_data
def loading_data(data="multi30k"):
    train_multi30k_df = load_data(data, default_file="train.csv", file_format="csv", as_numpy=True)
    vaild_multi30k_df = load_data(data, default_file="validation.csv", file_format="csv", as_numpy=True)
    test_multi30k_df = load_data(data, default_file="test.csv", file_format="csv", as_numpy=True)
    return train_multi30k_df, vaild_multi30k_df, test_multi30k_df
if __name__=="__main__":
    train_multi30k_df, vaild_multi30k_df, test_multi30k_df = loading_data()
    print("train_multi30k_df:")
    print(train_multi30k_df[:5])


# In[ ]:


## text embedding
import torch
import torch.nn as nn
import numpy as np

class TextBedding(nn.Module):
    def __init__(self,embedding_dim=128):
        super(TextBedding, self).__init__()
        self.embedding_dim=embedding_dim
        self.vocab = {}
        self.embedding_layer = nn.ModuleDict()  # Dictionary to hold embedding layers for different languages

    def tokenize(self,text):
        return text.lower().split()
    
    def build_vocab(self, data, language="English", vocab=None):
        if vocab is None:
            vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}  # Start with special tokens
        for text in data:
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.vocab[language] = vocab
        return vocab
    
    def initialize_embedding_layer(self,language="English"):
        if language not in self.vocab:
            raise ValueError(f"Vocabulary for language '{language}' must be built before initializing the embedding layer.")
        self.embedding_layer[language] = nn.Embedding(len(self.vocab[language]), self.embedding_dim)

    # Transform the whole data into 2D token IDs
    def text_to_ids(self, data, language="English", pad_token="<pad>"):
        add_sos_eos = True  # Add <sos> and <eos> tokens to each sequence
        if language not in self.vocab:
            raise ValueError(f"Vocabulary for language '{language}' is not built.")
        vocab = self.vocab[language]
        all_token_ids = []
        for text in data:
            tokens = self.tokenize(text)
            token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]  # Map tokens to IDs
            if add_sos_eos:
                token_ids = [vocab["<sos>"]] + token_ids + [vocab["<eos>"]]

            all_token_ids.append(token_ids)
    
        pad_token_id = vocab[pad_token]
        max_len = max(len(seq) for seq in all_token_ids)
        padded_token_ids = [
            seq + [pad_token_id] * (max_len - len(seq)) for seq in all_token_ids
        ]
        padded_token_ids_tensor = torch.tensor(padded_token_ids)
        return padded_token_ids_tensor

    def forward(self, token_ids, language="English"):
        if language not in self.embedding_layer:
            raise ValueError(f"Embedding layer for language '{language}' is not initialized. Call `initialize_embedding_layer(language='{language}')` first.")
        embeddings = self.embedding_layer[language](token_ids)
        return embeddings
    
if __name__ == "__main__":
    train_multi30k_df, vaild_multi30k_df, test_multi30k_df = loading_data()
    Embeddings_text=TextBedding()
    Embeddings_target=TextBedding()
    vocab_text = Embeddings_text.build_vocab(train_multi30k_df[:,0], language="English")  # Build vocab for source text
    vocab_target = Embeddings_target.build_vocab(train_multi30k_df[:,1], language="German")  # Build vocab for target text
    all_token_ids_text = Embeddings_text.text_to_ids(train_multi30k_df[:,0], language="English")
    all_token_ids_target = Embeddings_target.text_to_ids(train_multi30k_df[:,1], language="German")
    
    Embeddings_text.initialize_embedding_layer(language="English")
    embeddings_text = Embeddings_text.forward(all_token_ids_text, language="English")
    print("Token IDs:", all_token_ids_text[:5])  # Show the first 5 sequences of token IDs
    print("Embeddings shape:", embeddings_text.shape)  # Shape: (num_sequences, max_seq_len, embedding_dim)





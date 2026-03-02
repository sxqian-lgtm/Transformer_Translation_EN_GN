#!/usr/bin/env python
# coding: utf-8

# ## Description
# This notebook will implement the position embeddings. First, we will get the tokenlized input from the Text_processing.ipynb notebook. Then, we will implement the position embeddings and add them to the tokenlized input. Finally, we will visualize the position embeddings.

# In[1]:


import torch
import torch.nn as nn
from Text_Processing import TextBedding, loading_data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._create_pe(max_len)

    def _create_pe(self, max_len):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * -(torch.log(torch.tensor(10000.0)) / self.d_model))
        pe = torch.zeros(max_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            self._create_pe(seq_len)
            self.max_len = seq_len
        
        pe = self.pe[:seq_len].to(x.device, x.dtype)
        return x + pe.unsqueeze(0)


# In[2]:


if __name__ == "__main__":
    # Load data
    train_multi30k_df, vaild_multi30k_df, test_multi30k_df = loading_data()
    





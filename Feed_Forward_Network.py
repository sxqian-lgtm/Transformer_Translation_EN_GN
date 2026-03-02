#!/usr/bin/env python
# coding: utf-8

# ## Description  
# This notebook constructs the feed forward network part of the transformer architecture. We will transfer the output of the multi-head attention mechanism from Multi_Head_Attention.ipynb to this notebook and then we will implement the feed forward network by hands.

# In[3]:


import torch
import torch.nn as nn
from Text_Processing import TextBedding, loading_data
from Position_Encoding import PositionalEncoding
from Multi_Head_Attention import MultiHeadAttention
from torch.utils.data import DataLoader, TensorDataset


# In[ ]:


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        #d_model: Model dimension (same as input/output dimension of the attention layer)
        #d_ff: Dimension of the feed-forward layer (usually larger than d_model)
        super(FeedForward, self).__init__()
        self.W1=nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_model, d_ff)))  # First linear layer weight
        self.b1=nn.Parameter(torch.randn(d_ff))
        self.W2=nn.Parameter(nn.init.xavier_uniform_(torch.empty(d_ff, d_model)))  # Second linear layer weight
        self.b2=nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        x=torch.matmul(x, self.W1)+self.b1
        x=torch.relu(x)
        x=torch.matmul(x, self.W2)+self.b2
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # Learnable scaling factor
        self.beta = nn.Parameter(torch.zeros(d_model))  # Learnable bias
        self.epsilon = epsilon  # Small constant for numerical stability

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / (std + self.epsilon)  # Normalize the input
        return self.gamma * x_normalized + self.beta  # Scale and shift the normalized input
    
class ResnetBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super(ResnetBlock, self).__init__()
        self.feed_forward=FeedForward(d_model, d_ff)
        self.gamma=nn.Parameter(torch.ones(d_model))  # Learnable scaling factor for the residual connection
        self.beta=nn.Parameter(torch.zeros(d_model))  # Learnable bias for the residual connection
    def layer_norm(self,x,epsilon=1e-6):
        mean=x.mean(dim=-1, keepdim=True)
        std=x.std(dim=-1, keepdim=True, unbiased=False)
        x=self.gamma*(x-mean)/(std+epsilon)+self.beta   # Adding epsilon for numerical stability
        return x   
    
    def forward(self, x):
        x=self.layer_norm(x)
        x_residual=self.feed_forward(x)
        x=x+x_residual
        return x


# In[ ]:


if __name__ == "__main__":
    pass





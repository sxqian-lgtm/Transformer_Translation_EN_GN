#!/usr/bin/env python
# coding: utf-8

# ## Description
# This notebook constructs the most important part of the transformer architecture, the multi-head attention mechanism. We will transfer the tokenized text data with position encoding from Position_Encoding.ipynb to this notebook and then we will implement the multi-head attention mechanism by hands.  
# 

# In[ ]:


import torch
import torch.nn as nn
import math
from Text_Processing import TextBedding, loading_data
from Position_Encoding import PositionalEncoding
class MultiHeadAttention(nn.Module):
    def __init__(self,data_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert data_dim % num_heads == 0, "data_dim must be divisible by num_heads"
        self.data_dim = data_dim
        self.num_heads = num_heads
        self.head_dim = data_dim // num_heads
        self.WQ=nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.data_dim, self.data_dim)))  # Query weight matrix
        self.WK=nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.data_dim, self.data_dim)))  # Key weight matrix
        self.WV=nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.data_dim, self.data_dim)))  # Value weight matrix
        self.WO=nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.data_dim, self.data_dim)))  # Output weight matrix
        
    def Self_Attention(self, Q, K, V, mask=None):
        """
        Q: Query matrix, shape (batch_size, seq_len, head_dim)
        K: Key matrix, shape (batch_size, seq_len, head_dim)
        V: Value matrix, shape (batch_size, seq_len, head_dim)
        mask: Optional mask tensor for attention, 1 for valid positions and 0 for masked positions
        mask shape: (batch_size, seq_len, seq_len) for self-attention
        """
        d_k=Q.size(-1)  # head_dim
        scores=torch.matmul(Q,K.transpose(-2,-1))
        scores=scores/math.sqrt(d_k) #shape (batch_size, 1, seq_len_q, seq_len_k)
        if mask is not None: #shape (batch_size, 1, seq_len_q, seq_len_k)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attention_weights=torch.softmax(scores, dim=-1)
        attention_output=torch.matmul(attention_weights, V)
        return attention_output
        
    def Multi_Head_Attention(self, Q, K, V, mask=None):
        """
        Q: Query matrix, shape (batch_size, num_heads, seq_len, head_dim)
        K: Key matrix, shape (batch_size, num_heads, seq_len, head_dim)
        V: Value matrix, shape (batch_size, num_heads, seq_len, head_dim)
        mask: Optional mask tensor for attention, 1 for valid positions and 0 for masked positions
        when Q is from the decoder self-attention, mask shape: (batch_size, 1, seq_len_target, seq_len_target)
        when Q is from the decoder cross-attention, mask shape: (batch_size, 1, seq_len_target, seq_len_text)
        """
        d_k=Q.size(-1)  # head_dim
        scores=torch.matmul(Q,K.transpose(-2,-1))
        scores=scores/math.sqrt(d_k)
        if mask is not None:    #shape (batch_size, num_heads, seq_len_q, seq_len_k)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attention_weights=torch.softmax(scores, dim=-1)
        attention_output=torch.matmul(attention_weights, V)
        return attention_output
    def forward(self, x, Encoderoutput=None, mask=None):
        """
        x: (B, S_q, d_model)
        Encoderoutput: (B, S_k, d_model) or None
        mask: broadcastable to (B, num_heads, S_q, S_k) or (B,1,S_q,S_k)
        returns: (B, S_q, d_model)
        """
        B, S_q, _ = x.size()

        # Q projection and split heads -> (B, H, S_q, head_dim)
        q_proj = torch.matmul(x, self.WQ)
        q = q_proj.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V from encoder output if provided, otherwise from x (self-attention)
        if Encoderoutput is not None:
            _, S_k, _ = Encoderoutput.size()
            k_proj = torch.matmul(Encoderoutput, self.WK)
            v_proj = torch.matmul(Encoderoutput, self.WV)
            k = k_proj.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
            v = v_proj.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k_proj = torch.matmul(x, self.WK)
            v_proj = torch.matmul(x, self.WV)
            k = k_proj.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
            v = v_proj.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)

        # attention: expects (B, H, S_q, head_dim) inputs
        attn_out = self.Multi_Head_Attention(q, k, v, mask=mask)  # returns (B, H, S_q, head_dim)

        # combine heads -> (B, S_q, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S_q, self.data_dim)

        # final linear / projection
        out = torch.matmul(attn_out, self.WO)  # or self.WO(attn_out) if WO is nn.Linear
        return out


# In[6]:


from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    # Load and preprocess data
    train,vaild,test = loading_data()
    





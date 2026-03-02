#!/usr/bin/env python
# coding: utf-8

# ## Description
# This notebook focuses on constructing the Encoder and Decoder components of the Transformer architecture. We will derive the Multi-Head Attention mechanism and the Position-wise Feed-Forward Network from Feed_Forward_Network.ipynb, Multi_Head_Attention.ipynb, and Position_Encoding.ipynb.    

# In[ ]:


import torch
import torch.nn as nn
from Text_Processing import TextBedding, loading_data
from Position_Encoding import PositionalEncoding
from Multi_Head_Attention import MultiHeadAttention
from Feed_Forward_Network import ResnetBlock,LayerNorm,FeedForward
from torch.utils.data import DataLoader, TensorDataset


# In[ ]:


class EncoderBlock(nn.Module):
    def __init__(self, data_dim, num_heads, d_ff):
        # data_dim: Dimension of input and output features
        # num_heads: Number of attention heads
        # d_ff: Dimension of the feed-forward network
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(data_dim, num_heads)
        self.resnet = ResnetBlock(d_model=data_dim, d_ff=d_ff)
        self.layer_norm = LayerNorm(data_dim)  # Layer normalization encode attention output
    def forward(self, x, mask_encoder=None):
        # x: Input tensor, shape (batch_size, seq_len, data_dim)
        # mask_encoder: Optional mask for attention, shape (batch_size, seq_len, seq_len_encoder)
        x_raw=x
        x=self.layer_norm(x_raw)  # Layer normalization before attention
        attention_output = self.mha(x, mask=mask_encoder)  # Shape: (batch_size, seq_len, data_dim)
        output = attention_output + x_raw  # Residual connection + layer normalization
        output = self.resnet(output)  # Shape: (batch_size, seq_len, data_dim)
        return output
    
class DecoderBlock(nn.Module):
    def __init__(self, data_dim, num_heads, d_ff):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(data_dim, num_heads)  # Self-attention
        self.mha2 = MultiHeadAttention(data_dim, num_heads)  # Encoder-decoder attention
        self.resnet = ResnetBlock(d_model=data_dim, d_ff=d_ff)
        self.layer_norm1 = LayerNorm(data_dim)  # Layer normalization encode attention
        self.layer_norm2 = LayerNorm(data_dim)  # Layer normalization for encoder-decoder attention

    def forward(self, x, encoder_output, mask_decoder=None, mask_cross=None):
        # x: Input tensor for the decoder (target sequence), shape (batch_size, seq_len, data_dim)
        # encoder_output: Output from the encoder, shape (batch_size, seq_len_encoder, data_dim)
        # mask_decoder: Optional mask for decoder self-attention, shape (batch_size, seq_len, seq_len)
        # mask_cross: Optional mask for encoder-decoder attention, shape (batch_size, seq_len, seq_len_encoder)
        x_raw=x 
        x=self.layer_norm1(x_raw)  # Layer normalization before self-attention

        attention_output1 = self.mha1(x, mask=mask_decoder)  # Self-attention
        attention_output1 = attention_output1 + x_raw  # Residual connection + layer normalization
        # Encoder-decoder attention: Use encoder_output as keys and values, and attention_output1 as queries
        attention_output2_raw= attention_output1
        attention_output2 = self.layer_norm2(attention_output2_raw)  # Layer normalization before encoder-decoder attention
        attention_output2 = self.mha2(attention_output2, encoder_output, mask=mask_cross)  
        attention_output2 = attention_output2 + attention_output2_raw  # Residual connection + layer normalization
        output = self.resnet(attention_output2)  # Shape: (batch_size, seq_len, data_dim)
        return output


# In[ ]:


if __name__ == "__main__":
    # Load data
    train_multi30k_df, vaild_multi30k_df, test_multi30k_df = loading_data()
    Embeddings=TextBedding()
    vocab_text = Embeddings.build_vocab(train_multi30k_df[:,0])  # Build vocab for source text
    vocab_target = Embeddings.build_vocab(train_multi30k_df[:,1])  # Build vocab for target text
    all_token_ids_text = Embeddings.text_to_ids(train_multi30k_df[:,0], vocab_text)
    all_token_ids_target = Embeddings.text_to_ids(train_multi30k_df[:,1], vocab_target)

    vocab_size = len(vocab_text)  # Size of the vocabulary
    embedding_dim = 128  # Dimension of the embeddings
    embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    embeddings_text = embedding_layer(all_token_ids_text)  # Shape: (num_samples, seq_len, embedding_dim)

    dataset = TensorDataset(embeddings_text,all_token_ids_text, all_token_ids_target) 
    batch_size = 64 # Define your batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Iterate through batches
    for batch in dataloader:
        batch_embeddings = batch[0]  # Extract the embeddings from the batch,batch[1] and batch[2] are the token IDs for text and target respectively
        print("Batch shape:", batch_embeddings.shape)  # Shape: (batch_size, seq_len, embedding_dim)

        # Get sequence length and embedding dimension
        seq_len = batch_embeddings.shape[1]  # Assuming shape is (batch_size, seq_len, embedding_dim)
        d_model = batch_embeddings.shape[2]  # Assuming shape is (batch_size, seq_len, embedding_dim)

        # Apply positional encoding
        pos_encoder = PositionalEncoding(seq_len, d_model)
        pos_encoded_embeddings = pos_encoder.position_encoding(batch_embeddings)

        # You can now pass `pos_encoded_embeddings` to the encoder and decoder blocks as needed
        encoder_block = EncoderBlock(data_dim=d_model, num_heads=4, d_ff=256)
        encoder_output = encoder_block.forward(pos_encoded_embeddings)  # Shape: (batch_size, seq_len, d_model)
        decoder_block1 = DecoderBlock(data_dim=d_model, num_heads=4, d_ff=256)
        decoder_block2 = DecoderBlock(data_dim=d_model, num_heads=4, d_ff=256)
        decoder_output = decoder_block1.forward(pos_encoded_embeddings, encoder_output)  # Shape: (batch_size, seq_len, d_model)
        decoder_output = decoder_block2.forward(pos_encoded_embeddings, decoder_output)  # Shape: (batch_size, seq_len, d_model)

        print("Encoder output shape:", encoder_output.shape)  # Shape: (batch_size, seq_len, d_model)
        print("Decoder output shape:", decoder_output.shape)  # Shape: (batch_size, seq_len, d_model)
        



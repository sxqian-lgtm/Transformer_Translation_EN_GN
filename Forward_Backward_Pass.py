#!/usr/bin/env python
# coding: utf-8

# ## Description
# A notebook that demonstrates the forward and backward pass of the Transformer model, including how to compute gradients and update parameters during training. 

# In[1]:


import torch
import torch.nn as nn
from Text_Processing import TextBedding, loading_data
from Position_Encoding import PositionalEncoding
from Multi_Head_Attention import MultiHeadAttention
from Feed_Forward_Network import ResnetBlock
from torch.utils.data import DataLoader, TensorDataset
from Encoder_Decoder_Layers import EncoderBlock, DecoderBlock
from Utils import *


# In[ ]:


class ForwardBackwardPass(nn.Module):
    def __init__(self, embedding_train,output_size,d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, max_seq_length_text,max_seq_length_target):
        """embedded_train: the embedding layer initialized with the training data, 
           output_size: the size of the output vocabulary, 
           d_model: the dimension of the model, 
           nhead: the number of attention heads, 
           num_encoder_layers: the number of encoder layers, 
           num_decoder_layers: the number of decoder layers, 
           dim_feedforwad: the dimension of the feedforward network, 
           max_seq_length_text: the maximum sequence length for text positional encoding,
           max_seq_length_target: the maximum sequence length for target positional encoding.
        """
        super(ForwardBackwardPass, self).__init__()
        self.masking = Masking(padid=0)
        self.embedding = embedding_train
        self.positional_encoding_text = PositionalEncoding(d_model=d_model, max_len=max_seq_length_text)
        self.positional_encoding_target = PositionalEncoding(d_model=d_model, max_len=max_seq_length_target)
        self.encoder_layers=nn.ModuleList([EncoderBlock(d_model, nhead, dim_feedforward) for _ in range(num_encoder_layers)])
        self.decoder_layers=nn.ModuleList([DecoderBlock(d_model, nhead, dim_feedforward) for _ in range(num_decoder_layers)])
        self.output_layer =nn.Parameter(torch.randn(d_model, output_size, requires_grad=True))  # Randomly initialized output layer
        self.output_layer_bias = nn.Parameter(torch.zeros(output_size, requires_grad=True))  # Bias for the output layer
    def forward(self,textid,targetid,train_language="English",target_language="German"):
        # textid: [batch_size, seq_length]
        # targetid: [batch_size, seq_length]
        embedded_text = self.embedding[train_language](textid)  # [batch_size, seq_length, d_model]
        embedded_target = self.embedding[target_language](targetid)  # [batch_size, seq_length, d_model]
        pos_encoded_text = self.positional_encoding_text(embedded_text)  # [batch_size, seq_length, d_model]
        pos_encoded_target = self.positional_encoding_target(embedded_target)  # [batch_size, seq_length, d_model]

        # 3. Create masks
        
        src_mask = self.masking.create_padding_mask(textid)              # (B,1,1,S_text)
        tgt_mask = self.masking.create_decoder_mask(targetid)            # (B,1,S_tgt,S_tgt)
        cross_mask = self.masking.create_encoder_decoder_cross_mask(textid)  # (B,1,1,S_text)

        encoder_output = pos_encoded_text
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output,mask_encoder=src_mask)  # [batch_size, seq_length, d_model]
        decoder_output = pos_encoded_target
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, 
                                           mask_decoder=tgt_mask, mask_cross=cross_mask)  # [batch_size, seq_length, d_model]
        output = torch.matmul(decoder_output, self.output_layer) + self.output_layer_bias  # [batch_size, seq_length, vocab_size]
        return output
    def greedy_decode(self, src_ids, src_lang="English", tgt_lang="German",
                  sos_id=1, eos_id=2, max_len=50, device=None):
        # src_ids: (B, S_src)
        # src_lang: language of the source text (e.g., "English")
        # tgt_lang: language of the target text (e.g., "German")
        # sos_id: start-of-sequence token ID
        # eos_id: end-of-sequence token ID
        # max_len: maximum length of the generated sequence

        device = device or next(self.parameters()).device
        was_training = self.training
        self.eval()
        with torch.no_grad():
            src = src_ids.to(device)
            enc = self.positional_encoding_text(self.embedding[src_lang](src))
            src_mask = self.masking.create_padding_mask(src).to(device)

            memory = enc
            for layer in self.encoder_layers:
                memory = layer(memory, mask_encoder=src_mask)

            B = src.size(0)
            ys = torch.full((B,1), sos_id, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            for T in range(2,max_len+1):
                dec_in = self.positional_encoding_target(self.embedding[tgt_lang](ys)) # (B, T-1, d_model)
                tgt_mask = self.masking.create_decoder_mask(ys).to(device) # (B, 1, T-1, T-1)
                cross_mask = self.masking.create_encoder_decoder_cross_mask(src).to(device) # (B, 1, 1, S_src)
                print(f"shape: B={B}, T={T}, S_src={src.size(1)}, S_tgt={ys.size(1)}")

                out = dec_in
                for layer in self.decoder_layers:
                    out = layer(out, memory, mask_decoder=tgt_mask, mask_cross=cross_mask)

                logits = torch.matmul(out, self.output_layer) + self.output_layer_bias   # (B, T-1, vocab_size)
                print(f"logits shape: {logits.shape}")
                next_tok = logits[:, -1, :].argmax(dim=-1) # (B,)
                ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1) # (B, T)
                print(f"next_tok shape: {next_tok.shape}, ys shape: {ys.shape}")
                finished |= (next_tok == eos_id)
                if finished.all():
                    break

        if was_training:
            self.train()
        return ys


# In[5]:


if __name__ == "__main__":
    # Define hyperparameters
    embedding_dim = 128
    d_model = 128
    nhead = 4
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 1024
    max_seq_length_text = 37
    max_seq_length_target = 39
    # Load data
    train_multi30k_df, vaild_multi30k_df, test_multi30k_df = loading_data()
    
    Embed_train=TextBedding(embedding_dim=128)
    vocab_text = Embed_train.build_vocab(train_multi30k_df[:,0], language="English")  # Build vocab for source text
    vocab_target = Embed_train.build_vocab(train_multi30k_df[:,1], language="German")  # Build vocab for target text
    all_token_ids_text = Embed_train.text_to_ids(train_multi30k_df[:,0], language="English")
    all_token_ids_target = Embed_train.text_to_ids(train_multi30k_df[:,1], language="German")
    Embed_train.initialize_embedding_layer(language="English")
    Embed_train.initialize_embedding_layer(language="German")

    Transformer_model=ForwardBackwardPass(embedding_train=Embed_train.embedding_layer,
                                           output_size=len(vocab_target), d_model=embedding_dim, nhead=nhead, 
                                           num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
                                           dim_feedforward=dim_feedforward, max_seq_length_text=max_seq_length_text, max_seq_length_target=max_seq_length_target)
    

    dataset = TensorDataset(all_token_ids_text, all_token_ids_target) 
    batch_size = 64 # Define your batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Iterate through batches
    for batch in dataloader:
        textid_batch, targetid_batch = batch
        output = Transformer_model.forward(textid_batch, targetid_batch, train_language="English", target_language="German")
        print("Text ID Batch Shape:", textid_batch.shape)  # Shape: (batch_size, seq_len)
        print("Target ID Batch Shape:", targetid_batch.shape)  # Shape: (batch_size, seq_len)
        print("Output Shape:", output.shape)  # Shape: (batch_size, seq_len, vocab_size)
        break
        





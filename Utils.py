#!/usr/bin/env python
# coding: utf-8

# ## Description
# This notebook implements evaluation metrics commonly used for sequence-to-sequence tasks, such as BLEU score and ROUGE score, to assess the performance of the Transformer model. besides, it also includes mask generation functions for the Transformer model, such as padding masks and look-ahead masks, which are essential for training the model effectively

# In[ ]:


import torch
from matplotlib import pyplot as plt
class Masking:
    def __init__(self, padid: int = 0):
        """
        Convention: 1 = keep (visible), 0 = mask (blocked)
        padid: token id used for padding
        """
        self.padid = padid

    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, S_text) or (B, S_target), token ids
        return: (B, 1, 1, S)

        Description:
        Key-side padding visibility mask (1 = non-pad, 0 = pad)

        Used in:
        - Encoder self-attention: pass text
        - Cross-attention: pass text (K = text)
        - Decoder self-attention: used on the target side (combined with look-ahead mask)
        """
        device = seq.device
        keep = (seq != self.padid).to(device, dtype=torch.uint8)      # (B, S)
        return keep[:, None, None, :]                   # (B, 1, 1, S)

    def create_look_ahead_mask(self, seq_target: torch.Tensor) -> torch.Tensor:
        """
        return: (1, 1, S_tgt, S_tgt)

        Description:
        Causal mask for decoder self-attention.
        Lower triangle (including diagonal) = 1 (visible)
        Upper triangle = 0 (future tokens are blocked)
        # diagonal=0 : include diagonal in the visible part (allow attending to current token)
        """
        seq_len_target = seq_target.size(1)
        device = seq_target.device
        mask = torch.tril(
            torch.ones(seq_len_target, seq_len_target, device=device, dtype=torch.uint8),
            diagonal=0
        )[None, None, :, :]
        return mask

    def create_decoder_mask(self, seq_target: torch.Tensor) -> torch.Tensor:
        """
        Full decoder self-attention mask:

        (B, 1, S_tgt, S_tgt)
        = look-ahead(1,1,S,S)  &  target_key_padding(B,1,1,S)

        Unified semantics: 1 = visible, 0 = masked
        """
        B, S_tgt = seq_target.size()
        device = seq_target.device

        tgt_key_keep = self.create_padding_mask(seq_target).to(device, dtype=torch.uint8)  # (B,1,1,S)
        look_ahead   = self.create_look_ahead_mask(seq_target)                             # (1,1,S,S)

        # Logical AND with broadcasting → (B,1,S,S)
        # Visible only when both masks are 1
        combined = (look_ahead & tgt_key_keep)
        return combined.to(torch.uint8)

    def create_encoder_decoder_cross_mask(self, seq_text: torch.Tensor) -> torch.Tensor:
        """
        Key-side padding mask for Cross-Attention (Q = target, K = text).

        return: (B, 1, 1, S_text)
        """
        return self.create_padding_mask(seq_text).to(seq_text.device, dtype=torch.uint8)
    
class Loss_Plot:
    def __init__(self):
        self.train_loss_history = []
        self.val_loss_history = []

    def update(self, train_loss: float, val_loss: float):
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_history, label='Train Loss')
        plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.show()


# In[1]:


get_ipython().system('jupyter nbconvert --to script Utils.ipynb')


"""
Food to Recipe Model - Image to Recipe Generation
Uses SigLIP vision encoder and PyTorch transformer decoder
"""

import torch
import torch.nn as nn
import math
from transformers import SiglipVisionModel, AutoProcessor, AutoTokenizer
from typing import Optional


class FoodToRecipeModel(nn.Module):

    
    def __init__(
        self,
        siglip_model_name: str = "google/siglip-base-patch16-224",
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
        max_seq_len: int = 512,
        freeze_encoder: bool = False
    ):

        super().__init__()
        
        self.siglip_model_name = siglip_model_name
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        

        print(f"Loading SigLIP model: {siglip_model_name}")
        self.vision_encoder = SiglipVisionModel.from_pretrained(siglip_model_name)
        self.processor = AutoProcessor.from_pretrained(siglip_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(siglip_model_name)
        

        special_tokens = {}
        if self.tokenizer.pad_token is None:
            special_tokens['pad_token'] = '[PAD]'
        if self.tokenizer.bos_token is None:
            special_tokens['bos_token'] = '[BOS]'
        if self.tokenizer.eos_token is None:
            special_tokens['eos_token'] = '[EOS]'
        
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)
        
        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        

        if freeze_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        

        vision_hidden_size = self.vision_encoder.config.hidden_size

        self.vision_projection = nn.Linear(vision_hidden_size, d_model)
        

        self.token_embedding = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_token_id)
        

        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self._init_positional_encoding()
        
        self.dropout = nn.Dropout(dropout)
        

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        

        self.output_projection = nn.Linear(d_model, self.vocab_size)
        

        self._init_weights()
        
        print(f"Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def _init_positional_encoding(self):
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float) * 
                           -(math.log(10000.0) / self.d_model))
        
        pe = torch.zeros(1, self.max_seq_len, self.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.pos_encoding.data.copy_(pe)
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.vision_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state  # (batch, num_patches, hidden_size)
        
        image_features = self.vision_projection(vision_features)  # (batch, num_patches, d_model)
        
        return image_features
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        

        image_features = self.encode_image(pixel_values)
        

        token_embeds = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        token_embeds = token_embeds + self.pos_encoding[:, :seq_len, :]
        token_embeds = self.dropout(token_embeds)
        
        tgt_mask = self.generate_square_subsequent_mask(seq_len, device)
        
        decoder_output = self.transformer_decoder(
            tgt=token_embeds,
            memory=image_features,
            tgt_mask=tgt_mask
        )
        
        logits = self.output_projection(decoder_output)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95
    ) -> torch.Tensor:
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        

        image_features = self.encode_image(pixel_values)

        generated = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):

            logits = self.forward(pixel_values, generated)
            next_token_logits = logits[:, -1, :]
            

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            

            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = float('-inf')
            

            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            

            next_tokens = next_tokens.masked_fill(finished.unsqueeze(1), self.pad_token_id)
            

            generated = torch.cat([generated, next_tokens], dim=1)
            

            finished = finished | (next_tokens.squeeze(1) == self.eos_token_id)
            

            if finished.all():
                break
        
        return generated


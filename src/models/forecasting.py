import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    #Layers Definition
    def __init__(self,d_model:int,max_len : int=5000,dropout:float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position =torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0,d_model,2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len,1,d_model)
        pe[:,0,0::2] = torch.sin(position * div_term)
        pe[:,0,1::2] = torch.cos(position * div_term)
        self.register_buffer("pe",pe)
    #data flow
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ForecastingTransformer(nn.Module):
    def __init__(self,input_dim=3,d_model=64,nhead=4,num_layers=3,dim_feedforward=256,dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim,d_model)
        self.pos_encoder = PositionalEncoding(d_model,dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, input_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = x.transpose(0, 1)                                          # (seq_len, batch, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        seq_len = x.size(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(  # upper-triangular -inf mask
            seq_len, device=x.device, dtype=x.dtype
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        x = x[-1]                                                       # last token: (batch, d_model)
        return self.output_projection(x)                                # (batch, input_dim)
    



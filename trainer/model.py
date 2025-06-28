import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        return out

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        static_channels = 42
        self.static_cnn = nn.Sequential(
            nn.Conv2d(static_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResidualBlock(128, 256),
            ResidualBlock(256, 128),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        static_feature_dim = 64 * 4 * 9

        num_tiles = 34           # W, T, B (9*3=27) + F (4) + J (3) = 34
        d_model = 128
        nhead = 8
        num_encoder_layers = 2
        dim_feedforward = 256

        self.history_tile_embedding = nn.Embedding(num_tiles + 2, d_model, padding_idx=0)
        self.cls_token_id = num_tiles + 1
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=113)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True,
            dropout=0.1
        )
        self.history_transformer = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers)

        history_feature_dim = d_model
        fused_dim = static_feature_dim + history_feature_dim
        self.flatten = nn.Flatten()

        self._logits = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 235)
        )
        self._value_branch = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, TransformerEncoderLayer):
                if hasattr(m, 'self_attn') and hasattr(m.self_attn, 'in_proj_weight'):
                     nn.init.xavier_uniform_(m.self_attn.in_proj_weight)
                if hasattr(m, 'linear1') and hasattr(m.linear1, 'weight'):
                    nn.init.xavier_uniform_(m.linear1.weight)
                    if m.linear1.bias is not None: nn.init.zeros_(m.linear1.bias)
                if hasattr(m, 'linear2') and hasattr(m.linear2, 'weight'):
                    nn.init.xavier_uniform_(m.linear2.weight)
                    if m.linear2.bias is not None: nn.init.zeros_(m.linear2.bias)
        nn.init.normal_(self.history_tile_embedding.weight[self.cls_token_id], std=0.02)


    def forward(self, input_dict):
        obs = input_dict["observation"].float() # Shape: [B, 154, 4, 9]
        mask = input_dict["action_mask"].float()
        batch_size = obs.size(0)

        static_features_p1 = obs[:, :38]
        static_features_p2 = obs[:, 150:]
        static_input = torch.cat([static_features_p1, static_features_p2], dim=1)
        static_out = self.static_cnn(static_input)
        static_embedding = self.flatten(static_out)

        history_input = obs[:, 38:150] 
        history_ids = history_input[:, :, 0, 0].long() # Shape: [B, 112].

        cls_tokens = torch.full((batch_size, 1), self.cls_token_id, dtype=torch.long, device=obs.device)

        history_ids_with_cls = torch.cat([cls_tokens, history_ids], dim=1) # Shape: [B, 113]

        history_embedded = self.history_tile_embedding(history_ids_with_cls) # Shape: [B, 113, d_model]
        history_with_pos = self.pos_encoder(history_embedded)
        transformer_out = self.history_transformer(history_with_pos) # Shape: [B, 113, d_model]

        history_embedding = transformer_out[:, 0] # Shape: [B, d_model]

        fused_embedding = torch.cat([static_embedding, history_embedding], dim=1)

        logits = self._logits(fused_embedding)
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value = self._value_branch(fused_embedding)

        return masked_logits, value
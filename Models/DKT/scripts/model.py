import torch
import torch.nn as nn


class DKTLSTM(nn.Module):
    """
    Telemetry-Enriched Deep Knowledge Tracing using LSTM (extending Piech et al., NeurIPS 2015).
    Input: Interaction vector (2 * num_skills + telemetry_dim) combining concept-correctness
           one-hot encoding with bounded behavioral telemetry (latency, retries, scaffolds, mode).
    Output: Predicted probability of answering each skill correctly at the next step (num_skills).
    """
    def __init__(self, num_skills, telemetry_dim=13, hidden_dim=128, num_layers=1, dropout=0.2):
        super(DKTLSTM, self).__init__()
        self.num_skills = num_skills
        self.telemetry_dim = telemetry_dim
        self.hidden_dim = hidden_dim
        self.input_dim = 2 * num_skills + telemetry_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_skills)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 2 * num_skills + telemetry_dim)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        logits = self.fc(out)  # shape: (batch_size, seq_len, num_skills)
        preds = torch.sigmoid(logits)
        return preds, logits

    @torch.no_grad()
    def predict_mastery(self, x):
        """
        Live KnowledgeQuest inference helper: given a student's interaction history tensor
        of shape (1, seq_len, input_dim), returns the latest mastery vector (num_skills,).
        """
        was_training = self.training
        self.eval()
        preds, _ = self.forward(x)
        latest = preds[0, -1, :]
        if was_training:
            self.train()
        return latest


class DKTGRU(nn.Module):
    """
    Telemetry-Enriched Deep Knowledge Tracing using Gated Recurrent Units (GRU).
    Faster convergence and fewer parameters than LSTM, modeling cognitive decay and
    hesitation telemetry through coupled reset/update gates.
    """
    def __init__(self, num_skills, telemetry_dim=13, hidden_dim=128, num_layers=1, dropout=0.2):
        super(DKTGRU, self).__init__()
        self.num_skills = num_skills
        self.telemetry_dim = telemetry_dim
        self.hidden_dim = hidden_dim
        self.input_dim = 2 * num_skills + telemetry_dim

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_skills)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 2 * num_skills + telemetry_dim)
        out, _ = self.gru(x)
        out = self.dropout(out)
        logits = self.fc(out)
        preds = torch.sigmoid(logits)
        return preds, logits

    @torch.no_grad()
    def predict_mastery(self, x):
        """
        Live KnowledgeQuest inference helper: given a student's interaction history tensor
        of shape (1, seq_len, input_dim), returns the latest mastery vector (num_skills,).
        """
        was_training = self.training
        self.eval()
        preds, _ = self.forward(x)
        latest = preds[0, -1, :]
        if was_training:
            self.train()
        return latest


class SAKT(nn.Module):
    """
    Self-Attentive Knowledge Tracing (Pandey & Karypis, EDM 2019) with optional telemetry projection.
    Transformer-based attention model replacing recurrence with multi-head causal self-attention.
    """
    def __init__(self, num_skills, telemetry_dim=13, embed_dim=128, num_heads=4, num_layers=1, dropout=0.2, max_len=100):
        super(SAKT, self).__init__()
        self.num_skills = num_skills
        self.telemetry_dim = telemetry_dim
        self.embed_dim = embed_dim

        self.interaction_emb = nn.Embedding(2 * num_skills + 1, embed_dim, padding_idx=0)
        self.telemetry_proj = nn.Linear(telemetry_dim, embed_dim) if telemetry_dim > 0 else None
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_skills)
        self.dropout = nn.Dropout(dropout)

    def forward(self, interaction_indices, telemetry=None):
        # interaction_indices: (batch_size, seq_len) in range [1, 2*num_skills]
        seq_len = interaction_indices.size(1)
        positions = torch.arange(seq_len, device=interaction_indices.device).unsqueeze(0)

        # Causal mask so model only attends to past interactions
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=interaction_indices.device), diagonal=1).bool()

        x = self.interaction_emb(interaction_indices) + self.pos_emb(positions)
        if telemetry is not None and self.telemetry_proj is not None:
            x = x + self.telemetry_proj(telemetry)
        x = self.dropout(x)
        out = self.transformer_encoder(x, mask=causal_mask)
        logits = self.fc(out)
        preds = torch.sigmoid(logits)
        return preds, logits


class FinalDKTGRU(nn.Module):
    """
    Context-Aware, Regularized Deep GRU for Final Production Training (`FinalDKTGRU`).
    Combines:
      1. Dense interaction embeddings + telemetry + item difficulty at past step t,
      2. Multi-layer LayerNorm-regularized GRU memory state h_t,
      3. Target skill embedding + leak-free upcoming-question & spaced-repetition context at step t+1.
    """
    def __init__(
        self,
        num_skills,
        telemetry_dim=13,
        next_context_dim=9,
        embed_dim=96,
        hidden_dim=192,
        num_layers=2,
        dropout=0.3,
    ):
        super(FinalDKTGRU, self).__init__()
        self.num_skills = num_skills
        self.telemetry_dim = telemetry_dim
        self.next_context_dim = next_context_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 0 = padding, 1..M = skill answered right, M+1..2M = skill answered wrong
        self.inter_emb = nn.Embedding(2 * num_skills + 1, embed_dim, padding_idx=0)
        self.skill_emb = nn.Embedding(num_skills + 1, embed_dim, padding_idx=0)

        # Fuse past interaction embedding + past telemetry (13) + past item difficulty (1)
        step_in_dim = embed_dim + telemetry_dim + 1
        self.input_proj = nn.Sequential(
            nn.Linear(step_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.state_norm = nn.LayerNorm(hidden_dim)

        # Context-aware readout: [h_t, skill_emb(target_skill_{t+1}), next_context_{t+1}] -> P(correct_{t+1})
        head_in_dim = hidden_dim + embed_dim + next_context_dim
        self.readout = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inter_idx, past_telemetry, past_item_diff, target_skill_idx, next_context):
        """
        Args:
            inter_idx        : (B, T) long in [0, 2*num_skills]
            past_telemetry   : (B, T, telemetry_dim) float
            past_item_diff   : (B, T, 1) float
            target_skill_idx : (B, T) long in [0, num_skills] (1-indexed; 0 = pad)
            next_context     : (B, T, next_context_dim) float
        Returns:
            preds  : (B, T) probabilities in [0, 1]
            logits : (B, T) raw logits
        """
        e_inter = self.inter_emb(inter_idx)
        step_x = torch.cat([e_inter, past_telemetry, past_item_diff], dim=-1)
        step_h = self.input_proj(step_x)

        gru_out, _ = self.gru(step_h)
        h_t = self.state_norm(gru_out)

        e_target = self.skill_emb(target_skill_idx)
        joint = torch.cat([h_t, e_target, next_context], dim=-1)
        logits = self.readout(joint).squeeze(-1)
        preds = torch.sigmoid(logits)
        return preds, logits


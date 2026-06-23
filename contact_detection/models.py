"""Neural network models for binary contact detection.

논문 본류에서는 GRUDetector만 사용한다. 출력은 하나의 binary logit이고,
sigmoid(logit)이 P(contact)이다. contact region/localization head는 논문 범위에서
제외했기 때문에 제거했다.
"""

from __future__ import annotations

try:
    # torch.nn은 GRU, Linear, Dropout 같은 neural network layer를 제공한다.
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for contact_detection/models.py. "
        "Install the dependencies from contact_detection/requirements.txt."
    ) from exc


class GRUDetector(nn.Module):
    """GRU-based binary contact detector.

    입력 shape:
        [batch, window_length, input_dim]

    출력:
        [batch] binary logit

    logit은 아직 probability가 아니다. train/evaluate 단계에서
    BCEWithLogitsLoss 또는 sigmoid(logit)을 사용한다.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        gru_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        # GRU는 window 안의 시간 순서를 읽는 recurrent layer다.
        # batch_first=True라서 input shape이 [B, T, D]가 된다.
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=gru_dropout,
            bidirectional=bool(bidirectional),
            batch_first=True,
        )
        direction_multiplier = 2 if bidirectional else 1
        # 마지막 hidden state만 binary head로 보낸다.
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(hidden_dim) * direction_multiplier, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # outputs[:, -1, :]는 window 마지막 시점까지 본 GRU hidden representation이다.
        outputs, _ = self.gru(inputs)
        last_hidden = outputs[:, -1, :]
        # Linear head가 contact/no-contact logit 하나를 만든다.
        logits = self.head(self.dropout(last_hidden)).squeeze(-1)
        return logits


class MLPDetector(nn.Module):
    """Single-timestep learning baseline for binary contact detection.

    입력 shape:
        [batch, input_dim]

    optional compatibility:
        [batch, window_length, input_dim]가 들어오면 마지막 시점 feature만 사용한다.

    목적:
    - threshold보다 learning-based classification 자체가 유리한지 확인
    - GRU보다 temporal pattern이 실제로 추가 이득을 주는지 분리해서 비교
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        depth = max(1, int(num_layers))
        hidden = int(hidden_dim)
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _layer_idx in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            features = inputs[:, -1, :]
        else:
            features = inputs
        return self.net(features).squeeze(-1)

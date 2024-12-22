from torch import nn
import torch
import pytorch_lightning as pl

from typing import Optional
import numpy as np


def get_model(model_name):
    if model_name == "deeplob":
        return DeepLOB
    elif model_name == "transformer":
        return Transformer
    elif model_name == 'binbtabl':
        return BiN_BTABL
    elif model_name == 'binctabl':
        return BiN_CTABL
    elif model_name == 'lobtransformer':
        return LOBTransformer

    # You can add more model options here as needed
    raise ValueError(f"Unknown model name: {model_name}")



class DeepLOB(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.name = "deeplob"

        # Convolution blocks.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),  # (None, 100, 40, 1) -> (None, 100, 20, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 20, 32) -> (None, 100, 20, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 20, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),  # (None, 100, 10, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 10, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 10, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),  # (None, 100, 1, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 1, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 1, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # Inception modules.
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),  # (None, 100, 1, 64)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding="same"),  # (None, 100, 1, 64)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),  # (None, 100, 1, 64)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding="same"),  # (None, 100, 1, 64)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1,0)),  # (None, 100, 1, 32)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),  # (None, 100, 1, 64)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # LSTM layers.
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)  # (None, 64)
        self.fc1 = nn.Linear(64, 3)

    def forward(self, x):
    
        # Apply conv1
        x = self.conv1(x)

        # Apply conv2
        x = self.conv2(x)

        # Apply conv3
        x = self.conv3(x)

        # Apply inception modules
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)


        # Concatenate the inception module outputs
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        # Reshape for LSTM input
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        # LSTM processing
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Should be (batch_size, 64)

        # Fully connected layer
        logits = self.fc1(x)  # Output logits of shape (batch_size, 3)

        return logits




class LOBTransformer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.name = "lobtransformer"

        hidden = 32
        d_model = hidden * 2 * 3
        nhead = 8
        num_layers = 2 

        # Convolution blocks.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),  # (None, 100, 40, 1) -> (None, 100, 20, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 20, 32) -> (None, 100, 20, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 20, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),  # (None, 100, 10, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 10, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 10, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),  # (None, 100, 1, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 1, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),  # (None, 100, 1, 32)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # Inception modules.
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),  # (None, 100, 1, 64)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding="same"),  # (None, 100, 1, 64)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),  # (None, 100, 1, 64)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding="same"),  # (None, 100, 1, 64)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1,0)),  # (None, 100, 1, 32)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),  # (None, 100, 1, 64)
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cat_head = nn.Linear(d_model, 3)
        

    def forward(self, x):
    
        # Apply conv1
        x = self.conv1(x)

        # Apply conv2
        x = self.conv2(x)

        # Apply conv3
        x = self.conv3(x)

        # Apply inception modules
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)


        # Concatenate the inception module outputs
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        # Reshape for LSTM input
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        # LSTM processing
        x = self.transformer_encoder(x)
        # mean pool
        x = torch.mean(x, dim=1)

        logits = self.cat_head(x)

        return logits




class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(
        self, input_ids_shape: torch.Size, past_key_values_length: int = 0
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        _, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)




class Transformer(pl.LightningModule):
    def __init__(
        self,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
    ):
        super().__init__()
        self.name = "transformer"

        d_model = 64
        dim_feedforward = 256
        nhead = 8
        num_layers = 2

        self.embed = nn.Linear(40, d_model, bias=False)

        self.embed_positions = SinusoidalPositionalEmbedding(100, d_model)

        layer_norm_eps: float = 1e-5
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=encoder_norm
        )
        self.cat_head = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.embed(x.squeeze(1))

        embed_pos = self.embed_positions(x.shape)

        # transformer encoder
        x = self.transformer_encoder(x + embed_pos)

        # mean pool for classification
        x = torch.mean(x, dim=1)

        logits = self.cat_head(x)
        return logits



     

##########
## TABL ##
##########

class BiN(pl.LightningModule):
    def __init__(self, d2, d1, t1, t2): # d2:120, d1:40, t1:T, t2:5
        super().__init__()
        self.t1 = t1
        self.d1 = d1
        self.t2 = t2
        self.d2 = d2

        bias1 = torch.Tensor(t1, 1) # t1:T
        self.B1 = nn.Parameter(bias1)
        nn.init.constant_(self.B1, 0)

        l1 = torch.Tensor(t1, 1) # t1:T
        self.l1 = nn.Parameter(l1)
        nn.init.xavier_normal_(self.l1)

        bias2 = torch.Tensor(d1, 1) # d1:40
        self.B2 = nn.Parameter(bias2)
        nn.init.constant_(self.B2, 0)

        l2 = torch.Tensor(d1, 1) # d1:40
        self.l2 = nn.Parameter(l2)
        nn.init.xavier_normal_(self.l2)

        y1 = torch.Tensor(1, ) # weight X_time
        self.y1 = nn.Parameter(y1)
        nn.init.constant_(self.y1, 0.5)

        y2 = torch.Tensor(1, ) # weight X_space
        self.y2 = nn.Parameter(y2)
        nn.init.constant_(self.y2, 0.5)

    def forward(self, x):

        # if the two scalars are negative then we setting them to 0.01
        if (self.y1[0] < 0):
            y1 = torch.cuda.FloatTensor(1, )
            self.y1 = nn.Parameter(y1)
            nn.init.constant_(self.y1, 0.01)

        if (self.y2[0] < 0):
            y2 = torch.cuda.FloatTensor(1, )
            self.y2 = nn.Parameter(y2)
            nn.init.constant_(self.y2, 0.01)

        # normalization along the temporal dimension
        T2 = torch.ones([self.t1, 1], device="cuda") # (T,1)
        x2 = torch.mean(x, dim=2) # mean of each feature across sequence T
        x2 = torch.reshape(x2, (x2.shape[0], x2.shape[1], 1)) # (batch_size,40,1)

        std = torch.std(x, dim=2) # std of each feature across sequence T
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1)) # (batch_size,40,1)
        # it can be possible that the std of some temporal slices is 0, and this produces inf values, so we have to set them to one
        std[std < 1e-4] = 1

        diff = x - (x2 @ (T2.T))
        Z2 = diff / (std @ (T2.T)) # Z_time(batch_size,40,T)

        X2 = self.l2 @ T2.T # xavier_normal(40,1) @ ones(1,T)
        X2 = X2 * Z2 # for relative importance of Z_time rows across features/space
        X2 = X2 + (self.B2 @ T2.T) # (40,T)

        # normalization along the feature dimension
        T1 = torch.ones([self.d1, 1], device="cuda") # (40,1)
        x1 = torch.mean(x, dim=1) # mean at each time t of book features
        x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1], 1)) # (batch_size,T,1)

        std = torch.std(x, dim=1) # std at each time t of book features
        std = torch.reshape(std, (std.shape[0], std.shape[1], 1)) # (batch_size,T,1)

        op1 = x1 @ T1.T   # (batch_size,T,40)
        op1 = torch.permute(op1, (0, 2, 1))  # (batch_size,40,T)

        op2 = std @ T1.T
        op2 = torch.permute(op2, (0, 2, 1))  # (batch_size,40,T)

        z1 = (x - op1) / (op2) # Z_space(batch_size,40,T)
        X1 = (T1 @ self.l1.T) # ones(40,1) @ xavier_normal(1,T) 
        X1 = X1 * z1  # for relative importance of Z_space columns across time
        X1 = X1 + (T1 @ self.B1.T)

        # weighing the imporance of temporal and feature normalization
        x = self.y1 * X1 + self.y2 * X2

        return x
        


class BL_layer(pl.LightningModule): # d2:120, d1:40, t1:T, t2:5
    def __init__(self, d2, d1, t1, t2):
        super().__init__()
        weight1 = torch.Tensor(d2, d1)
        self.W1 = nn.Parameter(weight1)
        nn.init.kaiming_uniform_(self.W1, nonlinearity='relu') # (d2:120, d1:40)

        weight2 = torch.Tensor(t1, t2)
        self.W2 = nn.Parameter(weight2)
        nn.init.kaiming_uniform_(self.W2, nonlinearity='relu')  # (t1:T, t2:5)

        bias1 = torch.zeros((d2, t2))
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0) # (d2:120, t2:5)

        self.activation = nn.ReLU()

    def forward(self, x):
        # (self.W1 @ x) --> (120,40) @ (40,T) --> (120,T)
        # (x @ self.W2) --> (40,T) @ (T, 5) --> (40,5)

        x = self.activation(self.W1 @ x @ self.W2 + self.B) # (d2:120, t2:5)

        return x


        
class TABL_layer(pl.LightningModule):
    def __init__(self, d3, d2, t2, t3):  # d3(3), d2(120), t2(5), t3(1)
        super().__init__()
        self.t2 = t2

        weight = torch.Tensor(d3, d2) # (3,120)
        self.W1 = nn.Parameter(weight)
        nn.init.kaiming_uniform_(self.W1, nonlinearity='relu')

        weight2 = torch.Tensor(t2, t2) # (5,5)
        self.W = nn.Parameter(weight2)
        nn.init.constant_(self.W, 1 / t2)

        weight3 = torch.Tensor(t2, t3) # (5,1)
        self.W2 = nn.Parameter(weight3)
        nn.init.kaiming_uniform_(self.W2, nonlinearity='relu')

        bias1 = torch.Tensor(d3, t3) # (3,1)
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0)

        l = torch.Tensor(1, )
        self.l = nn.Parameter(l)
        nn.init.constant_(self.l, 0.5)

        self.activation = nn.ReLU()

    def forward(self, X):

        # maintaining the weight parameter between 0 and 1.
        if (self.l[0] < 0):
            l = torch.Tensor(1, )
            self.l = nn.Parameter(l)
            nn.init.constant_(self.l, 0.0)

        if (self.l[0] > 1):
            l = torch.Tensor(1, )
            self.l = nn.Parameter(l)
            nn.init.constant_(self.l, 1.0)

        # modelling the dependence along the first mode of X while keeping the temporal order intact (equation 7)
        X = self.W1 @ X    # (3,120) @ (120,5) -> (3,5)

        # enforcing constant (equation 1) on the diagonal
        W = self.W - self.W * torch.eye(self.t2, dtype=torch.float32, device="cuda") + torch.eye(self.t2, dtype=torch.float32, device="cuda") / self.t2

        # attention, the aim of the second step is to learn how important the temporal instances are to each other (equation 8)
        E = X @ W    # (3,5) @ (5,5) -> (3,5)

        # computing the attention mask A (equation 9)
        # the attention mask A obtained from the third step is used to zero out the effect of unimportant elements
        A = torch.softmax(E, dim=-1)   # (3,5)  relative importance of each time step within the compressed feature dimension

        # applying a soft attention mechanism (equation 10) over hard attention, otherwise hard attention can in  
        # early stages of the training select noisy information appearing to be most important ones  
        X = self.l[0] * (X) + (1.0 - self.l[0]) * X * A  # (3,5)

        # the final step of the proposed layer estimates the temporal mapping W2, after the bias shift (11)
        y = X @ self.W2 + self.B  # (3,5) @ (5,1) --> (3,1)
        return y        
        
        
        
        
class BiN_BTABL(pl.LightningModule):
    def __init__(self, d2, d1, t1, t2, d3, t3): # d2:120, d1:40, t1:T, t2:5, d3:3, t3:1
        super().__init__()

        self.name = 'binbtabl'

        self.BiN = BiN(d2, d1, t1, t2) # (40, T)
        self.BL = BL_layer(d2, d1, t1, t2)  # (d2:120, t2:5)
        self.TABL = TABL_layer(d3, d2, t2, t3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.squeeze(1)
        # first of all we pass the input to the BiN layer, then we use the B(TABL) architecture
        x = torch.permute(x, (0, 2, 1))

        x = self.BiN(x) # (40, T)

        self.max_norm_(self.BL.W1.data)
        self.max_norm_(self.BL.W2.data)
        x = self.BL(x) # (120, 5)
        x = self.dropout(x)

        self.max_norm_(self.TABL.W1.data)
        self.max_norm_(self.TABL.W.data)
        self.max_norm_(self.TABL.W2.data)
        x = self.TABL(x)
        x = torch.squeeze(x, 2)
        return x

    def max_norm_(self, w):
        with torch.no_grad():
            if (torch.linalg.matrix_norm(w) > 10.0):
                norm = torch.linalg.matrix_norm(w)
                desired = torch.clamp(norm, min=0.0, max=10.0)
                w *= (desired / (1e-8 + norm))


class BiN_CTABL(pl.LightningModule):
    def __init__(self, d2, d1, t1, t2, d3, t3, d4, t4):
        super().__init__()

        self.name = 'binctabl'

        self.BiN = BiN(d2, d1, t1, t2)
        self.BL = BL_layer(d2, d1, t1, t2)
        self.BL2 = BL_layer(d3, d2, t2, t3)
        self.TABL = TABL_layer(d4, d3, t3, t4)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.squeeze(1)
        # first of all we pass the input to the BiN layer, then we use the C(TABL) architecture
        x = torch.permute(x, (0, 2, 1))

        x = self.BiN(x)

        self.max_norm_(self.BL.W1.data)
        self.max_norm_(self.BL.W2.data)
        x = self.BL(x)
        x = self.dropout(x)

        self.max_norm_(self.BL2.W1.data)
        self.max_norm_(self.BL2.W2.data)
        x = self.BL2(x)
        x = self.dropout(x)

        self.max_norm_(self.TABL.W1.data)
        self.max_norm_(self.TABL.W.data)
        self.max_norm_(self.TABL.W2.data)
        x = self.TABL(x)
        x = torch.squeeze(x, 2)
        return x

    def max_norm_(self, w):
        with torch.no_grad():
            if (torch.linalg.matrix_norm(w) > 10.0):
                norm = torch.linalg.matrix_norm(w)
                desired = torch.clamp(norm, min=0.0, max=10.0)
                w *= (desired / (1e-8 + norm))
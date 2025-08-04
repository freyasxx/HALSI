import torch
import torch.nn as nn
import torch.nn.functional as F
from halsi.utils.general_utils import AttrDict
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def sinusoidal_positional_encoding(seq_len, dim, device):
    ''' Static positional encoding function '''
    position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-np.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (seq_len, dim)

class BaseVAE(nn.Module):
    def __init__(self, n_actions=9, n_obs=60, n_z=16,
                 n_hidden=128, n_layers=1, n_propr=530, device="cuda"):
        super().__init__()
        self.n_actions = n_actions
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_z = n_z
        self.device = device
        self.n_propr = n_propr

        self.bc_criterion = nn.MSELoss(reduction="mean")

        # ---------------------------Transformer encoder ---------------------------
        self.input_proj = nn.Linear(self.n_actions + self.n_obs, self.n_hidden)
        # Define CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.n_hidden))

        # Define Transformer encoder, using TransformerEncoderLayer, set nhead=4
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_hidden, nhead=4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        # ---------------------------------------------------------------------------
        
        self.encoder = nn.Sequential(nn.Linear(self.n_hidden, 64), 
                                     nn.BatchNorm1d(64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 32), 
                                     nn.BatchNorm1d(32),
                                     nn.LeakyReLU(),
                                     nn.Linear(32, self.n_z * 2))

        # Replace original decoder with diffusion decoder
        self.diffusion_decoder = DiffusionDecoder(n_obs=self.n_obs, n_z=self.n_z, n_actions=self.n_actions,
                                                  hidden_dim=64, T=25)   # T is the number of diffusion steps


    def init_hidden(self, batch_size):
        # For interface compatibility, init_hidden does not need to do anything here
        pass

    
    def run_inference(self, x, mask=None):
        """
        Use Transformer encoder to generate latent variables.
        Input:
            x: Tensor, shape (batch, seq_len, n_obs+n_actions)
            mask: Optional Bool Tensor, shape (batch, seq_len), where True indicates padding
        Output:
            Tensor, shape (batch, 2, n_z), first dim is mu, second is log_var
        """
        x = self.input_proj(x)      # (batch, seq_len, n_hidden)
        batch_size, seq_len, _ = x.shape

        # cls_token shape: (1, 1, n_hidden), expand to (batch, 1, n_hidden)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate [CLS] token to the front of the sequence, result shape: (batch, seq_len+1, n_hidden)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        pos_emb = sinusoidal_positional_encoding(seq_len + 1, self.n_hidden, x.device).unsqueeze(0)
        x = x + pos_emb
        # Handle mask: if mask is not None, add corresponding [CLS] token mask at the front (usually set to False)
        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # (batch, seq_len+1)
        else:
            mask = None

        x = x.transpose(0, 1)  # (seq_len+1, batch, n_hidden)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # (seq_len+1, batch, n_hidden)
        cls_out = x[0]
        out = self.encoder(cls_out)  # (batch, 2*n_z)
        return out.view(-1, 2, self.n_z)


    def run_decode_batch(self, x, seq_lens, fn):
        reconstructions = []
        for i, seq_len in enumerate(seq_lens):
            sample_x = x[i, :seq_len]
            actions = fn(sample_x)
            reconstructions.append(actions)
        return reconstructions

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def vae_loss(self, inputs, output, seq_lens, beta=0.00001):  # Initial: beta=0.00000001
        bc_losses = []
        for i, seq_len in enumerate(seq_lens):
            rec = output.reconstruction[i]
            act = inputs["actions"][i][:seq_len]
            # Create mask
            mask = torch.ones_like(act)
            # For variable-length sequences, rec may need to be truncated to match act length
            rec = rec[:seq_len]
            # Compute reconstruction loss, use mask to ensure invalid positions are not included
            bc_loss = self.bc_criterion(rec * mask, act * mask)
            bc_losses.append(bc_loss)

        bc_loss = torch.mean(torch.stack(bc_losses))
        log_var = torch.clamp(output.q.log_var, min=-20, max=2)
        kld_loss = (-0.5 * torch.sum(1 + log_var - output.q.mu.pow(2) - log_var.exp())) * beta
        return bc_loss, kld_loss
    
    
    def loss(self, inputs, output, seq_lens):

        bc_loss, kld_loss = self.vae_loss(inputs, output, seq_lens)
        total_loss = bc_loss + kld_loss

        return AttrDict(bc_loss=bc_loss,
                        kld_loss=kld_loss,
                        total_loss=total_loss)
    
class DiffusionDecoder(nn.Module):
    """
    Diffusion generative model, used to convert conditional information (state and latent variable) into action generation.
    """
    def __init__(self, n_obs, n_z, n_actions, hidden_dim=64, T=100):
        super().__init__()
        self.T = T  # Number of diffusion steps
        self.n_obs = n_obs
        self.n_z = n_z
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        # Define beta schedule, linearly increasing
        self.beta = torch.linspace(1e-4, 0.02, T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # Define denoising network, input includes: noisy actions, conditional info (state and latent variable), and timestep embedding
        self.net = nn.Sequential(
            nn.Linear(n_actions + n_obs + n_z + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x, condition, t):
        """
        x: actions under current noise, shape: (batch, n_actions)
        condition: conditional info, shape: (batch, n_obs+n_z)
        t: current timestep (scalar or tensor), expand and concatenate
        """
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float, device=x.device)
        t_emb = t.expand(x.size(0), 1)  # timestep embedding
        input_net = torch.cat([x, condition, t_emb], dim=1)
        return self.net(input_net)

    def sample(self, condition, seq_lens):
        """
        Conditional info condition: (batch, seq_len, n_obs+n_z (41))
        seq_lens: List[int], actual length of each sequence
        Returns: actions, shape (batch, seq_len, n_actions)
        """

        batch_size, max_seq_len, condition_dim = condition.shape
        device = condition.device

        x = torch.randn(batch_size, max_seq_len, self.n_actions, device=device)
 

        for t in reversed(range(self.T)):
            t_tensor = torch.full((batch_size, max_seq_len, 1), t, device=device, dtype=torch.float)
            input_net = torch.cat([x, condition, t_tensor], dim=-1)  # (batch, seq_len, dim)

            # Reshape for MLP input: (batch * seq_len, dim)
            input_net_flat = input_net.view(-1, input_net.shape[-1])
            
            eps_pred = self.net(input_net_flat).view(batch_size, max_seq_len, self.n_actions)

            beta_t = self.beta[t].to(device)
            alpha_t = self.alpha[t].to(device)
            alpha_bar_t = self.alpha_bar[t].to(device)

            x = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred)

            if t > 0:
                noise = torch.randn_like(x)
                x = x + noise * torch.sqrt(beta_t)

        # Use seq_lens to construct mask, zero out padding regions
        mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < torch.tensor(seq_lens, device=device).unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        x = x * mask

        return x
    

class dynamicsVAE(BaseVAE):
    def forward(self, x, seq_lens):
        states = x['obs']  
        actions = x['actions']  

        # Encode
        self.init_hidden(len(states))
        x_cat = []
        for i, seq_len in enumerate(seq_lens):
            state = states[i][:seq_len]  
            action = actions[i][:seq_len]  

            x_cat.append(torch.cat((state, action), dim=1))  
        
        # Pad variable-length sequences to get (batch, max_seq_len, n_obs+n_actions)
        x_cat = nn.utils.rnn.pad_sequence(x_cat, batch_first=True)
        x = self.run_inference(x_cat)  
        q = AttrDict(mu=x[:, 0, :], log_var=x[:, 1, :])
        z = self.reparameterize(q.mu, q.log_var)

        # Decode
        decode_inputs = []
        for i, seq_len in enumerate(seq_lens):
            state = states[i][:seq_len] 
            z_tiled = z[i].unsqueeze(0).repeat(seq_len, 1)  
            decode_inputs.append(torch.cat((state, z_tiled), dim=1))  
        
        # Pad and decode
        decode_inputs = nn.utils.rnn.pad_sequence(decode_inputs, batch_first=True)
       
        reconstruction = self.diffusion_decoder.sample(decode_inputs, seq_lens)
        
        return AttrDict(reconstruction=reconstruction, q=q, z=z)
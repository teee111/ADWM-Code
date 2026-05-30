import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_1d_sincos_pos_embed(embed_dim, length):
    if embed_dim % 2 != 0:
        raise ValueError("Embed dim must be divisible by 2")

    pos = torch.arange(length, dtype=torch.float32)
    grid = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (grid / (embed_dim // 2)))

    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.unsqueeze(0)

# --- Time Embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = emb.to(dtype=x.dtype)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class VLMFeatureAdapter(nn.Module):
    def __init__(self, vlm_feat_dim, hidden_dim, num_queries=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        
        self.input_norm = nn.LayerNorm(vlm_feat_dim)
        self.feature_proj = nn.Linear(vlm_feat_dim, hidden_dim)
        
        # Learnable Queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True, 
            activation="gelu",
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, vlm_features):
        """
        vlm_features: (B, N, vlm_feat_dim)
        Returns: (B, num_queries, hidden_dim)
        """
        B = vlm_features.shape[0]
        
        memory = self.input_norm(vlm_features)       # (B, N, vlm_feat_dim)
        memory = self.feature_proj(memory)            # (B, N, hidden_dim)
        
        tgt = self.query_embed.expand(B, -1, -1)
        
        out = self.transformer_decoder(tgt, memory)
        return out



class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # 6 params: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, emb_t, attn_mask=None):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = \
            self.adaLN_modulation(emb_t).chunk(6, dim=1)

        # Self-Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn1(
            x_norm, x_norm, x_norm, 
            attn_mask=attn_mask, 
            need_weights=False
        )[0]

        # MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x

        

class VLAModel(nn.Module):
    def __init__(self, 
                 action_dim=7, 
                 proprio_dim=7, 
                 hidden_dim=384, 
                 num_heads=6, 
                 depth=12, 
                 action_len=32, 
                 proprio_len=8, 
                 num_registers=2,
                 vlm_feat_dim=384,     
                 dino_feat_dim=1024,
                 vlm_num_queries=64,   
                 adapter_depth=2,
                 use_dino=False,
                 use_internal_guidance=False,  
                 ig_layer=4,   
                 state_inject_start=0,  
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_len = action_len
        self.proprio_len = proprio_len
        self.num_registers = num_registers
        self.use_dino = use_dino
        self.use_internal_guidance = use_internal_guidance
        self.ig_layer = ig_layer
        
        self.state_inject_start = state_inject_start

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        
        self.register_buffer('action_pos_emb', get_1d_sincos_pos_embed(hidden_dim, action_len))
        self.register_buffer('proprio_pos_emb', get_1d_sincos_pos_embed(hidden_dim, proprio_len))
        
        self.type_emb_action = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_emb_proprio = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.type_emb_vlm = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        if self.use_dino:
            self.type_emb_dino = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        nn.init.normal_(self.type_emb_action, std=0.02)
        nn.init.normal_(self.type_emb_proprio, std=0.02)
        nn.init.normal_(self.type_emb_vlm, std=0.02)

        # --- Register Tokens ---
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_registers, hidden_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # --- Vision Adapters ---
        self.vlm_visual_adapter = VLMFeatureAdapter(
            vlm_feat_dim=vlm_feat_dim, hidden_dim=hidden_dim, num_queries=vlm_num_queries,
            num_heads=num_heads, num_layers=adapter_depth, dropout=0.
        )
        if self.use_dino:
            self.dino_adapter = VLMFeatureAdapter(
                vlm_feat_dim=dino_feat_dim, hidden_dim=hidden_dim, num_queries=vlm_num_queries,
                num_heads=num_heads, num_layers=adapter_depth, dropout=0.
            )

        # --- Transformer Blocks ---
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads) for _ in range(depth)])

        # --- Head (IG) ---
        if self.use_internal_guidance:
            self.intermediate_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            self.intermediate_output_proj = nn.Linear(hidden_dim, action_dim)

        # --- Output Head ---
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        self.gamma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    
    def forward(self,
                t, 
                noisy_actions, 
                qpos_history=None,
                vlm_features=None,
                dino_features=None,
                obs_tokens_evolved=None,
                ):
        if obs_tokens_evolved is not None:
            return self._forward_with_evolved_obs(t, noisy_actions, obs_tokens_evolved)
        
        t_emb = self.time_mlp(t)
        
        x_action = self.action_proj(noisy_actions) + \
                   self.action_pos_emb[:, :noisy_actions.shape[1], :] + \
                   self.type_emb_action
                   
        x_proprio_real = self.proprio_proj(qpos_history) + \
                         self.proprio_pos_emb[:, :qpos_history.shape[1], :] + \
                         self.type_emb_proprio

        x_proprio = torch.zeros_like(x_proprio_real)

        tokens_list = [x_action, x_proprio]
        current_len = x_action.shape[1]
        
        proprio_start = current_len
        proprio_end = proprio_start + x_proprio.shape[1]
        current_len = proprio_end

        if self.num_registers > 0:
            regs = self.register_tokens.expand(noisy_actions.shape[0], -1, -1)
            tokens_list.append(regs)
            current_len += regs.shape[1]
            
        if vlm_features is not None:
            cond_vlm = self.vlm_visual_adapter(vlm_features) + self.type_emb_vlm
            tokens_list.append(cond_vlm) 
            if self.use_dino and dino_features is not None:
                cond_dino = self.dino_adapter(dino_features) + self.type_emb_dino
                tokens_list.append(cond_dino)

        x = torch.cat(tokens_list, dim=1)
        
        intermediate_pred = None
        state_injected = False  
        
        for i, block in enumerate(self.blocks):
            
            if i == self.state_inject_start and not state_injected:
                x[:, proprio_start:proprio_end, :] = x_proprio_real
                state_injected = True
            x = block(x, t_emb)
            
            if self.use_internal_guidance and i == (self.ig_layer - 1):
                x_inter = self.intermediate_norm(x)
                x_action_inter = x_inter[:, :self.action_len, :]
                intermediate_pred = self.intermediate_output_proj(x_action_inter)
        
        x = self.final_norm(x)
        x_action_out = x[:, :self.action_len, :]
        final_pred = self.output_proj(x_action_out)
        
        gamma_logit = self.gamma_head(x_action_out[:, 0, :])
        gamma = torch.sigmoid(gamma_logit + 3.0) 
        obs_tokens_out = x[:, self.action_len:, :]

        out = {
            "final_pred": final_pred,
            "gamma": gamma,
            "obs_tokens_evolved": obs_tokens_out, 
        }
        
        if self.use_internal_guidance:
            out["intermediate_pred"] = intermediate_pred
            
        return out


    def _forward_with_evolved_obs(self, t, noisy_actions, obs_tokens_evolved):
        t_emb = self.time_mlp(t)
        
        # noisy action tokens
        x_action = self.action_proj(noisy_actions) + \
                   self.action_pos_emb[:, :noisy_actions.shape[1], :] + \
                   self.type_emb_action
        
        # Concat [new_action_tokens, evolved_obs_tokens]
        x = torch.cat([x_action, obs_tokens_evolved], dim=1)
        
        for block in self.blocks:
            x = block(x, t_emb)
        
        x = self.final_norm(x)
        x_action_out = x[:, :self.action_len, :]
        final_pred = self.output_proj(x_action_out)
        
        return {"final_pred": final_pred}




def calc_flow_matching_loss_ig(
    model, 
    x1, 
    vlm_features, 
    qpos_history, 
    qpos_target=None,
    dino_features=None,
    time_sampler="uniform",
    time_mu=0.0,      
    time_sigma=1.0,    
    lambda_ig=0.5,
    use_velocity_weighting=False,
    alpha=0.2,         
    sigma=0.01,        
    use_gamma_weighting=False,
    lambda_gamma=0.1,
    # Next-Chunk Prediction 
    x1_next=None,                
    use_next_chunk_pred=False,     # use next-chunk prediction
    lambda_next_chunk=0.5,         # next-chunk loss weight
):
    """
    Flow Matching Loss (IG + Next-Chunk Prediction)
    """
    device = x1.device
    bs = x1.shape[0]
    action_len = x1.shape[1] 
    
    x0 = torch.randn_like(x1)
    
    if time_sampler == "uniform":
        t = torch.rand(bs, device=device)
    elif time_sampler == "logit_normal":
        normal_samples = torch.randn(bs, device=device)
        normal_samples = normal_samples * time_sigma + time_mu
        t = torch.sigmoid(normal_samples)
    else:
        raise ValueError(f"Unsupported time_sampler: {time_sampler}")
    
    t_expand = t.view(bs, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * x1
    
    target_v = x1 - x0 
    
    preds = model(t, 
                  noisy_actions=x_t, 
                  vlm_features=vlm_features, 
                  dino_features=dino_features,
                  qpos_history=qpos_history)
    
    pred_v_final = preds["final_pred"]
    gamma = preds["gamma"]
    pred_v_inter = preds.get("intermediate_pred", None)
    obs_tokens_evolved = preds["obs_tokens_evolved"]  # (B, N_obs, hidden_dim)

    if use_velocity_weighting:
        velocities = torch.diff(x1[:,:,:-1], dim=1) 
        mean_vel = torch.mean(torch.abs(velocities), dim=(1, 2)) 
        weights = alpha + (1.0 - alpha) * torch.exp(-(mean_vel ** 2) / (2 * sigma ** 2))
    else:
        weights = torch.ones(bs, device=device)
        mean_vel = torch.zeros(bs, device=device)
    
    if use_gamma_weighting:
        steps = torch.arange(action_len, device=device, dtype=torch.float32).view(1, -1, 1)
        temporal_weights = torch.pow(gamma.unsqueeze(2), steps)
        loss_reg = -lambda_gamma * gamma.mean()
    else:
        temporal_weights = 1.0
        loss_reg = 0.0

    loss_final_unreduced = F.mse_loss(pred_v_final, target_v, reduction='none')
    
    loss_final_unreduced = loss_final_unreduced * temporal_weights
    
    loss_final_per_sample = torch.mean(loss_final_unreduced, dim=(1, 2)) 
    loss_final = torch.mean(loss_final_per_sample * weights)
    
    # Intermediate Loss
    if pred_v_inter is not None:
        loss_inter_unreduced = F.mse_loss(pred_v_inter, target_v, reduction='none')
        loss_inter_unreduced = loss_inter_unreduced * temporal_weights
        
        loss_inter_per_sample = torch.mean(loss_inter_unreduced, dim=(1, 2))
        loss_inter = torch.mean(loss_inter_per_sample * weights)
        
        loss_mse = loss_final + lambda_ig * loss_inter
    else:
        loss_mse = loss_final

    # =====================================================================
    # Next-Chunk Prediction Loss
    # =====================================================================
    loss_next_chunk = torch.tensor(0.0, device=device)
    
    if use_next_chunk_pred and x1_next is not None:
        x0_next = torch.randn_like(x1_next)
        
        if time_sampler == "uniform":
            t_next = torch.rand(bs, device=device)
        elif time_sampler == "logit_normal":
            normal_samples_next = torch.randn(bs, device=device)
            normal_samples_next = normal_samples_next * time_sigma + time_mu
            t_next = torch.sigmoid(normal_samples_next)
        else:
            t_next = torch.rand(bs, device=device)
        
        t_next_expand = t_next.view(bs, 1, 1)
        x_t_next = (1 - t_next_expand) * x0_next + t_next_expand * x1_next
        target_v_next = x1_next - x0_next
        
        frozen_params = {k: v.detach() for k, v in model.named_parameters()}
        
        preds_next = torch.func.functional_call(
            model,
            frozen_params,
            args=(t_next,),
            kwargs={
                'noisy_actions': x_t_next,
                'obs_tokens_evolved': obs_tokens_evolved, 
            }
        )
        
        pred_v_next = preds_next["final_pred"]
        
        # --- next-chunk flow matching loss ---
        loss_next_unreduced = F.mse_loss(pred_v_next, target_v_next, reduction='none')
        loss_next_chunk = torch.mean(loss_next_unreduced)
    loss = loss_mse + loss_reg + lambda_next_chunk * loss_next_chunk

    return loss, {
        "pred_v": pred_v_final, 
        "target_v": target_v,
        "mean_chunk_vel": mean_vel.mean().item(),
        "mean_loss_weight": weights.mean().item(),
        "mean_gamma": gamma.mean().item(),
        "loss_mse": loss_mse.item() if isinstance(loss_mse, torch.Tensor) else loss_mse,
        "loss_reg": loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg,
        "loss_next_chunk": loss_next_chunk.item() if isinstance(loss_next_chunk, torch.Tensor) else loss_next_chunk,
    }
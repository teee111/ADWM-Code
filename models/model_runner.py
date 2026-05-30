import torch
import torch.nn as nn
import logging
import json
from dataclasses import dataclass, field
from typing import Optional
from transformers import Qwen3VLForConditionalGeneration

from .vla_model_fm import VLAModel, calc_flow_matching_loss_ig


logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def create_vlm(checkpoint_path, dtype=torch.bfloat16, device="cuda", vlm_feat_layer=-4):
        logger.info(f"Loading Frozen VLM from {checkpoint_path}...")
        
        vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=False,
            attn_implementation="sdpa"
        )
            
        vlm_model.lm_head = nn.Identity()
        
        if vlm_feat_layer < 0:
            total_layers = len(vlm_model.model.language_model.layers)
            keep_layers = total_layers + vlm_feat_layer + 1
            removed = total_layers - keep_layers
            vlm_model.model.language_model.layers = vlm_model.model.language_model.layers[:keep_layers]
            logger.info(f"Truncated VLM language_model: {total_layers} -> {keep_layers} layers (removed last {removed})")
        
        vlm_model.eval()
        for param in vlm_model.parameters():
            param.requires_grad = False
        
        return vlm_model

    @staticmethod
    def create_action_model(config, vlm_hidden_size, use_ig=False, ig_layer=4):
        logger.info("Initializing unified VLAModel (Early Fusion)...")
        model = VLAModel(
            action_dim=config.common.action_dim,
            proprio_dim=config.common.state_dim,
            vlm_feat_dim=vlm_hidden_size,       
            hidden_dim=config.model.action_expert.hidden_size, 
            action_len=config.common.action_chunk_size,
            proprio_len=config.common.proprio_len if hasattr(config.common, 'proprio_len') else 1, 
            depth=config.model.action_expert.depth,
            num_heads=config.model.action_expert.num_heads,
            vlm_num_queries=config.model.action_expert.vlm_adapter_num_queries,
            use_dino=config.model.use_vision_encoder_feat,
            dino_feat_dim=config.model.dino_feat_dim,
            use_internal_guidance=use_ig,
            ig_layer=ig_layer,
            num_registers=config.model.num_registers,
            state_inject_start=0,
        )
        return model



class VLAWrapper(nn.Module):
    def __init__(self,
                 vlm_model, 
                 action_model, 
                 time_sampler, 
                 visual_feat_layer,
                 vlm_feat_layer,
                 stage1_mode,
                 device, 
                 dtype, 
                 norm_stats_path,
                 use_ig=False,
                 train_config = None
                 ):
        super().__init__()
        self.vlm = vlm_model
        self.action_model = action_model
        self.time_sampler = time_sampler
        self.visual_feat_layer = visual_feat_layer
        self.vlm_feat_layer = vlm_feat_layer
        
        self.stage1_mode = stage1_mode
        self.device = device
        self.dtype = dtype

        self.use_ig = use_ig
        
        
        self.train_config = train_config
        
        if self.train_config is not None:
            self.ig_lambda = self.train_config.ig_lambda
            self.use_vel_weight = self.train_config.use_vel_weight
            self.vel_weight_alpha = self.train_config.vel_weight_alpha
            self.vel_weight_sigma = self.train_config.vel_weight_sigma
            self.use_time_decay_weight = self.train_config.use_time_decay_weight
            self.time_decay_weight_gamma = self.train_config.time_decay_weight_gamma
            self.use_next_chunk_pred = getattr(self.train_config, 'use_next_chunk_pred', False)
            self.lambda_next_chunk = getattr(self.train_config, 'lambda_next_chunk', 0.5)
        else:
            self.use_next_chunk_pred = False
            self.lambda_next_chunk = 0.5
        
        self.stage1_mode = (self.vlm is None)
        logger.info(f"VLAWrapper initialized in {'Stage 1 (No VLM)' if self.stage1_mode else 'Stage 2'} mode.")
        if self.use_next_chunk_pred:
            logger.info(f"Next-Chunk Prediction ENABLED (lambda={self.lambda_next_chunk})")

        self.load_norm_stats(norm_stats_path)
        
        self._visual_blocks_output = None
        if self.vlm is not None:
            last_block = self.vlm.model.visual.blocks[self.visual_feat_layer]
            last_block.register_forward_hook(self._visual_hook_fn)
            logger.info("Registered forward hook on VLM visual blocks.")


    def load_norm_stats(self, path):
        logger.info(f"Loading normalization stats from {path}...")
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            stats = data.get('robotwin2')
            action_stats = stats.get('action')
            state_stats = stats.get('state')

            if action_stats:
                act_min = torch.tensor(action_stats['min'], dtype=torch.float32)
                act_max = torch.tensor(action_stats['max'], dtype=torch.float32)
                self.register_buffer('action_min', act_min)
                self.register_buffer('action_max', act_max)
                logger.info(f"Loaded Action stats - Dim: {len(action_stats['min'])}")
            
            if state_stats:
                state_min = torch.tensor(state_stats['min'], dtype=torch.float32)
                state_max = torch.tensor(state_stats['max'], dtype=torch.float32)
                self.register_buffer('state_min', state_min)
                self.register_buffer('state_max', state_max)
                logger.info(f"Loaded State stats - Dim: {len(state_stats['min'])}")
            
        except Exception as e:
            logger.error(f"Failed to load normalization stats: {e}")
            raise e

    def _visual_hook_fn(self, module, input, output):
        self._visual_blocks_output = output

    def get_vlm_features(self, vlm_inputs, vlm_feat_layer):
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in vlm_inputs.items()}
        
        with torch.no_grad():
            inputs_embeds = self.vlm.get_input_embeddings()(inputs['input_ids'])
            
            image_embeds, _ = self.vlm.get_image_features(
                inputs['pixel_values'], inputs['image_grid_thw']
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)
            
            image_mask, _ = self.vlm.model.get_placeholder_mask(
                inputs['input_ids'], inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
            position_ids, _ = self.vlm.model.get_rope_index(
                input_ids=inputs['input_ids'],
                image_grid_thw=inputs['image_grid_thw'],
                video_grid_thw=None,
                attention_mask=inputs['attention_mask']
            )

            vlm_output = self.vlm.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs['attention_mask'],
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True
            )
            vlm_latent_feat = vlm_output.hidden_states[vlm_feat_layer]
            return vlm_latent_feat

    def _normalize_tensor(self, x, min_val, max_val):
        min_v = min_val.to(device=x.device, dtype=x.dtype)
        max_v = max_val.to(device=x.device, dtype=x.dtype)
        
        denominator = max_v - min_v
        denominator[denominator < 1e-6] = 1.0 
        
        norm_x = 2 * (x - min_v) / denominator - 1
        return norm_x

    def normalize_action(self, action):
        return self._normalize_tensor(action, self.action_min, self.action_max)

    def normalize_state(self, state):
        return self._normalize_tensor(state, self.state_min, self.state_max)

    def denormalize_action(self, norm_action):
        action_min = self.action_min.to(device=norm_action.device, dtype=norm_action.dtype)
        action_max = self.action_max.to(device=norm_action.device, dtype=norm_action.dtype)
        
        denominator = action_max - action_min
        denominator[denominator < 1e-6] = 1.0
        
        action = (norm_action + 1) / 2 * denominator + action_min
        return action

    def forward(self, batch):
        vlm_feats = None
        if not self.stage1_mode and batch.get('vlm_inputs') is not None:
            vlm_feats = self.get_vlm_features(batch['vlm_inputs'], self.vlm_feat_layer)

        x1_raw = batch['action_sequence'].to(self.device, self.dtype)
        qpos_raw = batch['state'].to(self.device, self.dtype)
        
        if qpos_raw.dim() == 2: 
            qpos_raw = qpos_raw.unsqueeze(1) 
        
        x1 = self.normalize_action(x1_raw)  
        qpos = self.normalize_state(qpos_raw) 
        
        if self.stage1_mode:
            qpos_history = qpos[:, :-1, : ]
            visual_feats = None
        else:
            qpos_history = qpos[:, :-1, : ]
        
            visual_feats = self._visual_blocks_output.detach()
            batch_size = vlm_feats.shape[0]
            C = visual_feats.shape[-1]
            visual_feats = visual_feats.view(batch_size, -1, C)
        
        x1_next = None
        if self.use_next_chunk_pred and batch.get('next_action_sequence') is not None:
            x1_next_raw = batch['next_action_sequence'].to(self.device, self.dtype)
            x1_next = self.normalize_action(x1_next_raw)
            
        loss, info_dic = calc_flow_matching_loss_ig(
            self.action_model, 
            x1=x1, 
            vlm_features=vlm_feats, 
            dino_features=visual_feats,
            qpos_history=qpos_history,
            time_sampler=self.time_sampler,
            lambda_ig=self.ig_lambda,
            
            use_velocity_weighting=self.use_vel_weight,
            alpha=self.vel_weight_alpha,         
            sigma=self.vel_weight_sigma,        
            use_gamma_weighting=self.use_time_decay_weight,
            lambda_gamma=self.time_decay_weight_gamma,
            
            x1_next=x1_next,
            use_next_chunk_pred=self.use_next_chunk_pred,
            lambda_next_chunk=self.lambda_next_chunk,
        )
        
        return loss, info_dic
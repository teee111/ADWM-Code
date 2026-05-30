import os
import h5py
import torch
import numpy as np
from datetime import datetime
import functools
import torch.nn as nn



class AttentionRecorder:
    def __init__(self, action_model: torch.nn.Module, run_dir: str): 
        self.model = action_model
        

        self.base_dir = run_dir 
        
        self.task_name = "default"
        self.rollout_idx = 0
        self.infer_idx = 0
        self.denoise_idx = 0
        
        self.current_maps = []
        self.hooks = []
        self.num_blocks = 0
        
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if 'blocks' in name and name.endswith('attn1') and isinstance(module, torch.nn.MultiheadAttention):
                handle = module.register_forward_hook(self._hook_fn)
                self.hooks.append(handle)
                self.num_blocks += 1
                
        print(f"[*] AttentionRecorder: Successfully registered hooks on {self.num_blocks} DiT blocks.")

    def _hook_fn(self, module, input, output):
        attn_weights = output[1]  # (B, L, S)
        
        if attn_weights is not None:
            attention_matrix = attn_weights[0].detach().float().cpu().numpy()
            
            self.current_maps.append(attention_matrix)
        
        if len(self.current_maps) == self.num_blocks and self.num_blocks > 0:
            self._save_and_clear()

    def set_context(self, task_name: str, rollout_idx: int, infer_idx: int):
        self.task_name = task_name
        self.rollout_idx = rollout_idx
        self.infer_idx = infer_idx
        self.denoise_idx = 0 

    def _save_and_clear(self):
        save_dir = os.path.join(self.base_dir, self.task_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # {rollout_idx}_{infer_idx}_{denoise_idx}.hdf5
        filename = f"{self.rollout_idx}_{self.infer_idx}_{self.denoise_idx}.hdf5"
        filepath = os.path.join(save_dir, filename)
        
        # (num_blocks, seq_len, seq_len)
        stacked_maps = np.stack(self.current_maps, axis=0)
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('attention_maps', data=stacked_maps, compression="gzip")
        
        self.current_maps.clear()
        self.denoise_idx += 1

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()



class AttentionRecorderAdapter:
    def __init__(self, action_model, run_dir, h_tokens=16, w_tokens=20):
        self.model = action_model
        self.run_dir = run_dir
        
        self.h_tokens = h_tokens
        self.w_tokens = w_tokens
        
        self.L_a = self.model.action_len
        self.L_p = self.model.proprio_len
        self.L_r = self.model.num_registers
        self.L_vlm = self.model.vlm_visual_adapter.num_queries
        
        if hasattr(self.model, 'dino_adapter') and self.model.use_dino:
            self.L_dino = self.model.dino_adapter.num_queries
            self.start_dino = self.L_a + self.L_p + self.L_r + self.L_vlm
        else:
            raise ValueError("model not vision encoder feature")

        self.dit_records = {}
        self.adapter_records = {}
        self.hooks = []
        
        self.task_name = None
        self.rollout_idx = 0
        self.infer_idx = 0
        self.denoise_idx = 0  
        
        self._patch_mha_safely()
        self._register_model_hook()

    def set_context(self, task_name, rollout_idx, infer_idx):
        self.task_name = task_name
        self.rollout_idx = rollout_idx
        self.infer_idx = infer_idx
        self.denoise_idx = 0  
        self.clear_records()

    def _wrap_mha(self, layer_name, module, record_dict):
        old_forward = module.forward

        @functools.wraps(old_forward)
        def new_forward(*args, **kwargs):
            fast_kwargs = kwargs.copy()
            fast_kwargs['need_weights'] = False
            out_fast = old_forward(*args, **fast_kwargs)

            with torch.no_grad():
                slow_kwargs = kwargs.copy()
                slow_kwargs['need_weights'] = True
                slow_kwargs['average_attn_weights'] = True
                _, weights = old_forward(*args, **slow_kwargs)
                
                if weights is not None:
                    record_dict[layer_name] = weights.detach().float().cpu()

            return out_fast

        module.forward = new_forward

    def _patch_mha_safely(self):
        for i, block in enumerate(self.model.blocks):
            self._wrap_mha(f"layer_{i}", block.attn1, self.dit_records)

        for i, layer in enumerate(self.model.dino_adapter.transformer_decoder.layers):
            self._wrap_mha(f"adapter_layer_{i}", layer.multihead_attn, self.adapter_records)

    def _register_model_hook(self):
        def model_post_forward_hook(mod, inp, out):
            if self.task_name is not None:
                self.save_and_map()
                self.denoise_idx += 1  
                self.clear_records()   
        self.hooks.append(self.model.register_forward_hook(model_post_forward_hook))

    def clear_records(self):
        self.dit_records.clear()
        self.adapter_records.clear()

    def save_and_map(self):
        if len(self.dit_records) == 0 or len(self.adapter_records) == 0:
            return
            
        save_dir = os.path.join(self.run_dir, self.task_name)
        os.makedirs(save_dir, exist_ok=True)
        
        file_name = f"episode_{self.rollout_idx}_{self.infer_idx}_{self.denoise_idx}.hdf5"
        save_path = os.path.join(save_dir, file_name)
        
        with h5py.File(save_path, 'w') as f:
            grp_adapter = f.create_group('adapter_raw')
            for k, v in self.adapter_records.items():
                grp_adapter.create_dataset(k, data=v.numpy())
                
            grp_dit = f.create_group('dit_raw')
            for k, v in self.dit_records.items():
                grp_dit.create_dataset(k, data=v.numpy())

            grp_mapped = f.create_group('mapped_2d')

            avg_adapter_attn = torch.stack(list(self.adapter_records.values()), dim=0).mean(dim=0)

            for dit_layer_name, dit_attn in self.dit_records.items():

                attn_a2c = dit_attn[:, :self.L_a, self.start_dino : self.start_dino + self.L_dino]
                
                # (B, 32, 64) x (B, 64, 512) -> (B, 32, 512)
                mapped_1d = torch.bmm(attn_a2c, avg_adapter_attn)
                
                try:
                    mapped_2d = mapped_1d.view(-1, self.L_a, self.h_tokens, self.w_tokens)
                    grp_mapped.create_dataset(f"{dit_layer_name}_mapped", data=mapped_2d.numpy())
                except RuntimeError as e:
                    print(f"⚠️ Reshape Fail: {mapped_1d.shape[-1]}")
                    raise e

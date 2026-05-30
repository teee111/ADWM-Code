import os
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from omegaconf import OmegaConf
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_lsq_spline


from models.model_runner import ModelFactory, VLAWrapper
from transformers import AutoProcessor
from utils.vlm_utils import preprocess_vlm_messages


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def bspline_smooth(action_seq: np.ndarray, degree: int = 3, num_ctrl_pts: int = 8) -> np.ndarray:
    """
    B-Spline filter
    """
    N, D = action_seq.shape
    
    if N <= num_ctrl_pts:
        return action_seq

    x = np.arange(N)
    
    num_internal_knots = num_ctrl_pts - degree
    
    internal_knots = np.linspace(0, N - 1, num_internal_knots + 2)[1:-1]
    
    knots = np.concatenate([
        [0] * (degree + 1), 
        internal_knots, 
        [N - 1] * (degree + 1)
    ])
    
    spline = make_lsq_spline(x, action_seq, knots, k=degree)
    
    smoothed_action = spline(x)
    
    return smoothed_action

class RobotWinInference:

    def __init__(
        self, 
        config_path: str, 
        checkpoint_path: str, 
        norm_stats_path: str,
        # two_stage_mode: bool,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        smooth_actions: bool = False,  
        smooth_sigma: float = 1.0,     # gauss filter
        use_ig: bool = False,          # use Internal Guidance
        ig_scale: float = 1.5,         # IG scale for inference
        ig_layer: int = 4,             # IG internal layer 
        ig_stop_step: int = 10        
    ):
        self.device = device
        self.dtype = dtype
        self.smooth_actions = smooth_actions
        self.smooth_sigma = smooth_sigma

        self.use_ig = use_ig
        self.ig_scale = ig_scale
        self.ig_layer = ig_layer
        
        self.ig_stop_step = ig_stop_step
        
        
        self.action_queue = deque()
        # load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        self.config = OmegaConf.load(config_path)
        
        self.num_inference_steps = self.config.common.num_inference_steps     # Flow Matching 步数
        self.action_execution_horizon = self.config.common.action_execution_horizon
        
        time_sampler = self.config.training.time_sampler
        # init VLM Processor
        vlm_ckpt = self.config.model.vlm.checkpoint_path
        logger.info(f"Loading VLM Processor form {vlm_ckpt}...")

        self.vlm_processor = AutoProcessor.from_pretrained(vlm_ckpt, trust_remote_code=False)
        
        logger.info(f"Loading Model from {checkpoint_path}...")
        
        vlm_model = ModelFactory.create_vlm(
            self.config.model.vlm.checkpoint_path,
            dtype,
            device,
            vlm_feat_layer=self.config.model.vlm_feat_layer
            )
        vlm_hidden_size = vlm_model.config.text_config.hidden_size
        
        action_model = ModelFactory.create_action_model(
            self.config,
            vlm_hidden_size=vlm_hidden_size,
            use_ig=self.use_ig,
            ig_layer=self.ig_layer
        )
        
        self.model = VLAWrapper(
            vlm_model=vlm_model,
            action_model=action_model,
            time_sampler=time_sampler,
            visual_feat_layer=self.config.model.visual_feat_layer,
            vlm_feat_layer=self.config.model.vlm_feat_layer,
            stage1_mode=False,
            device=device,
            dtype=dtype,
            norm_stats_path=norm_stats_path,
            use_ig=self.use_ig  
        )
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        # print('state_dict:', state_dict)
        

        msg = self.model.action_model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded Action Model weights. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
        
        self.model.eval()
        self.model.to(device, dtype)
        
        indices_config = self.config.dataset.indices_config
        
        # TODO
        self.processor = RobotWinInferenceProcessor(
            vlm_processor=self.vlm_processor,
            indices_config=indices_config,
            camera_name=self.config.dataset.get('camera_names', ['head_camera'])[0],
            device=device,
            dtype=dtype
        )

        # Hook
        visual_feat_layer = self.config.model.visual_feat_layer
        self._visual_blocks_output = None
        if vlm_model is not None:
            last_block = vlm_model.model.visual.blocks[visual_feat_layer]
            last_block.register_forward_hook(self._visual_hook_fn)
            logger.info("Registered forward hook on VLM visual blocks.")

        
        logger.info("Inference Engine Ready.")

    def _visual_hook_fn(self, module, input, output):
        self._visual_blocks_output = output

    def reset(self):
        """buffer clean"""
        self.processor.reset()
        self.action_queue.clear()  
        
    @torch.no_grad()
    def _predict_chunk(self, observation: Dict[str, Any], instruction: str = "") -> np.ndarray:
        
        batch = self.processor.process(observation, instruction)
        
        qpos_cond = self.model.normalize_state(batch['state']) 

        vlm_feats = self.model.get_vlm_features(batch['vlm_inputs'], self.config.model.vlm_feat_layer)
        
        # Flow Matching Sampling
        B = 1
        action_len = self.config.common.action_chunk_size
        action_dim = self.config.common.action_dim
        
        # Init Noise
        x_t = torch.randn((B, action_len, action_dim), device=self.device, dtype=self.dtype)
        steps = torch.linspace(0, 1, self.num_inference_steps + 1, device=self.device, dtype=self.dtype)
        
        
        
        visual_feats = self._visual_blocks_output.detach()  # torch.Size([40960, 1024])
        # [Total_Tokens, C] >>> [B, N, C]
        batch_size = vlm_feats.shape[0]
        C = visual_feats.shape[-1]  # 1024
        visual_feats = visual_feats.view(batch_size, -1, C)
        
        
        # ODE Solver
        for i in range(self.num_inference_steps):
            t_curr = steps[i]
            dt = steps[i+1] - t_curr
            t_input = t_curr.unsqueeze(0)
            
            preds = self.model.action_model(
                t=t_input,
                noisy_actions=x_t,
                qpos_history=qpos_cond,
                vlm_features=vlm_feats,
                dino_features=visual_feats
            )
            
            pred_v_final = preds["final_pred"]
            pred_v_inter = preds.get("intermediate_pred", None)
            
            # Internal Guidance 
            if self.use_ig and pred_v_inter is not None:
                
                # D_w = D_i + w * (D_f - D_i)
                if i < self.ig_stop_step:
                    pred_v = pred_v_inter + self.ig_scale * (pred_v_final - pred_v_inter)
                else:
                    pred_v = pred_v_final
            else:
                # not use IG
                pred_v = pred_v_final

            x_t = x_t + pred_v * dt
            
        # Denormalize
        action_seq = self.model.denormalize_action(x_t) # [1, 32, 14]
        
        action_np = action_seq[0].float().cpu().numpy()  # [32, 14]
        
        if self.smooth_actions:
            # TODO
            # gauss
            action_np = gaussian_filter1d(action_np, sigma=self.smooth_sigma, axis=0, radius=5)
            
            # bspline
            # action_np = bspline_smooth(action_np, degree=3, num_ctrl_pts=8)
            
        return action_np
    
    
    
    

    def step(self, observation: Dict[str, Any], instruction: str = "") -> np.ndarray:
        """
         Receding Horizon Control
        """

        self.processor.update_state_buffer(observation)
        
        
        if len(self.action_queue) == 0:

            full_chunk = self._predict_chunk(observation, instruction) # [32, 14]
            
            valid_actions = full_chunk[:self.action_execution_horizon] # [N, 14]
            
            for act in valid_actions:
                self.action_queue.append(act)
        
        if len(self.action_queue) > 0:
             pass

        action = self.action_queue.popleft()
        
        return action



class RobotWinInferenceProcessor:
    def __init__(
        self, 
        vlm_processor: Any, 
        indices_config: Dict[str, List[int]] = None,
        camera_name: str = 'head_camera',
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.processor = vlm_processor
        self.device = device
        self.dtype = dtype
        self.camera_name = camera_name
        
        self.state_indices = indices_config['state_indices']
        
        self.history_len = 1 + abs(min(self.state_indices))
        
        self.state_buffer = deque(maxlen=self.history_len)
        
    def reset(self):
        self.state_buffer.clear()


    def update_state_buffer(self, observation: Dict[str, Any]):
        current_state = self._parse_state_from_obs(observation)
        
        if len(self.state_buffer) == 0:
            for _ in range(self.history_len):
                self.state_buffer.append(current_state)
        else:
            self.state_buffer.append(current_state)


    def _parse_state_from_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        endpose = obs['endpose']
        
        l_pose = np.array(endpose['left_endpose'], dtype=np.float32)
        l_grip = np.array([endpose['left_gripper']], dtype=np.float32)
        r_pose = np.array(endpose['right_endpose'], dtype=np.float32)
        r_grip = np.array([endpose['right_gripper']], dtype=np.float32)
        
        # Concatenate (16,)
        state_vec = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=0)
        return state_vec

    def process(self, observation: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        state_seq_np = np.stack(list(self.state_buffer), axis=0)
        
        # trans to  [1, T_state, D]
        state_tensor = torch.from_numpy(state_seq_np).to(self.device, self.dtype).unsqueeze(0)
        
        if self.camera_name in observation['observation']:
            img_np = observation['observation'][self.camera_name]['rgb']
            
            pil_image = Image.fromarray(img_np)
            
            vlm_inputs_single = preprocess_vlm_messages(instruction, [pil_image], self.processor)
            
            vlm_inputs = {
                'input_ids': vlm_inputs_single['input_ids'].to(self.device),
                'pixel_values': vlm_inputs_single['pixel_values'].to(self.device, self.dtype),
                'attention_mask': vlm_inputs_single['attention_mask'].to(self.device),
                'image_grid_thw': vlm_inputs_single.get('image_grid_thw').to(self.device) if vlm_inputs_single.get('image_grid_thw') is not None else None
            }
        else:
            logger.warning(f"Camera {self.camera_name} not found in observation!")
        
        batch = {
            'state': state_tensor,          # [1, T, D]
            'vlm_inputs': vlm_inputs,       # Dict with tensors on device
        }
        
        return batch
    
    
class ActionRecorder:
    def __init__(self):
        self.actions = []

    def record(self, action):
        if hasattr(action, 'cpu'):
            action = action.cpu().detach().numpy()
        if action.ndim > 1:
            action = action.squeeze(0)
        self.actions.append(action)

    def plot_and_save(self, save_dir, episode_id):
        if not self.actions:
            return
        
        actions_np = np.array(self.actions) # shape: (T, 14)
        T, D = actions_np.shape
        
        fig, axes = plt.subplots(7, 2, figsize=(15, 20))
        axes = axes.flatten()
        
        for d in range(min(D, 14)):
            axes[d].plot(actions_np[:, d], color='b')
            axes[d].set_title(f'Action Dimension {d} (Joint Angle)')
            axes[d].set_xlabel('Step')
            axes[d].set_ylabel('Value')
            axes[d].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'action_episode_{episode_id}.png')
        plt.savefig(save_path)
        plt.close(fig)
        
        self.actions = []
    
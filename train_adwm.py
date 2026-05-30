import os
import sys
import torch
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass

from hdf5_dataloader.dataset import RobotWinTaskDataset, collate_fn, create_dataset

from utils.train_utils import inspect_batch_stats, count_parameters


from models.model_runner import ModelFactory, VLAWrapper


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    return config


class LossLogger:
    def __init__(self, log_dir="log/loss"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(log_dir, f"train_loss_{timestamp}.csv")
        with open(self.log_file, 'w') as f:
            f.write("Epoch,Step,Global_Step,Loss\n")

    def log(self, epoch, step, global_step, loss):
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{step},{global_step},{loss:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLA Two-Stage Training Script")
    
    parser.add_argument("--config", type=str, default="./configs/robotwin8.yaml", help="Path to config file")
    parser.add_argument("--norm_stats_path", type=str, default="./utils/stat-200-10.json", help="Path to normalization stats")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vla", help="Directory to save checkpoints")
    
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")

    parser.add_argument("--stage1", default=False, help="Enable Stage 1 (Motor Babbling) mode")
    
    # for stage 1 training
    parser.add_argument("--use_target_qpos", default=False, help="Enable Target qpos training")
    
    # for state 2 training 
    parser.add_argument("--stage1_checkpoint", type=str, 
                        default=None, 
                        help="Path to Stage 1 checkpoint (Required for Stage 2)"
                        )

    # RESUME
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from "
                             "(restores model + optimizer + scheduler + epoch)")

    # --- IG ---
    parser.add_argument("--use_ig", type=bool, default=True, help="Enable Internal Guidance auxiliary loss")
    parser.add_argument("--ig_layer", type=int, default=4, help="Layer index for intermediate supervision")
    parser.add_argument("--ig_lambda", type=float, default=0., help="Weight for the intermediate loss")
    
    # --- Next-Chunk Prediction ---
    parser.add_argument("--use_next_chunk_pred", type=bool, default=False, 
                        help="Enable next-chunk prediction loss for obs token supervision")
    parser.add_argument("--lambda_next_chunk", type=float, default=0.3, 
                        help="Weight for the next-chunk prediction loss")
    
    args = parser.parse_args()
    
    @dataclass
    class TrainConfig:
        ig_lambda: float = args.ig_lambda
        use_vel_weight: bool = False
        vel_weight_alpha: float = 0.2
        vel_weight_sigma: float = 0.01
        use_time_decay_weight: bool = False
        time_decay_weight_gamma: float = 0.05
        use_next_chunk_pred: bool = args.use_next_chunk_pred
        lambda_next_chunk: float = args.lambda_next_chunk
    
    train_cfg = TrainConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    logger.info(f"Using device: {device}, Precision: {dtype}")
    logger.info(f"Next-Chunk Prediction: {'ENABLED' if args.use_next_chunk_pred else 'DISABLED'} "
                f"(lambda={args.lambda_next_chunk})")
    
    config = load_config(args.config)
    
    if args.stage1:
        batch_size = config.training.batch_size_stage1
    else:
        batch_size = config.training.batch_size

    logger.info(f"Creating Dataset (Stage 1 Mode: {args.stage1})...")
    
    train_dataset = create_dataset(config, val=False, stage1_mode=args.stage1, 
                                   use_next_chunk=args.use_next_chunk_pred)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.system.num_workers, 
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    logger.info(f"Dataset Size: {len(train_dataset)} samples/episodes")
    logger.info(f"Batches per Epoch: {len(train_dataloader)}")

    if args.stage1:
        logger.info(">>> Initializing STAGE 1: Motor Babbling (No VLM)")
        
        vlm_model = None
        vlm_hidden_size = 2048
        
        action_model = ModelFactory.create_action_model(config, vlm_hidden_size, use_ig=args.use_ig, ig_layer=args.ig_layer)
        
    else:
        logger.info(">>> Initializing STAGE 2: Instruction Following (With VLM)")
        
        vlm_model = ModelFactory.create_vlm(
            config.model.vlm.checkpoint_path,
            dtype,
            device,
            vlm_feat_layer=config.model.vlm_feat_layer
            )
        vlm_hidden_size = vlm_model.config.text_config.hidden_size
        
        action_model = ModelFactory.create_action_model(config, vlm_hidden_size, use_ig=args.use_ig, ig_layer=args.ig_layer)

        if args.stage1_checkpoint and not args.resume:
            logger.info(f"Loading pre-trained Stage 1 weights from {args.stage1_checkpoint}")
            checkpoint = torch.load(args.stage1_checkpoint, map_location='cpu', weights_only=False)
            state_dict = checkpoint['model_state_dict']
            
            msg = action_model.load_state_dict(state_dict, strict=False)
            logger.info(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        elif args.stage1_checkpoint and args.resume:
            logger.warning("Both --resume and --stage1_checkpoint set. "
                           "Ignoring --stage1_checkpoint; will load weights from --resume.")
        elif not args.resume:
            logger.warning("No Stage 1 checkpoint provided! Training from scratch.")

    action_model.to(device, dtype=dtype)
    action_model.train()
    
    count_parameters(action_model, model_name="Action Model (Trainable)")
    
    
    model = VLAWrapper(
        vlm_model=vlm_model,
        action_model=action_model,
        time_sampler=config.training.time_sampler,
        visual_feat_layer=config.model.visual_feat_layer,
        vlm_feat_layer=config.model.vlm_feat_layer,
        stage1_mode=args.stage1,
        device=device,
        dtype=dtype,
        norm_stats_path=args.norm_stats_path,
        use_ig=args.use_ig,
        train_config=train_cfg
    )
    
    
    current_lr = config.training.learning_rate

    optimizer = AdamW(
        action_model.parameters(),
        lr=current_lr,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay
    )
    

    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(train_dataloader), 
        eta_min=5e-5
    )

    start_epoch = 0
    global_step = 0

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        logger.info(f">>> Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)

        saved_stage = ckpt.get('stage', None)
        expected_stage = 1 if args.stage1 else 2
        if saved_stage is not None and saved_stage != expected_stage:
            raise ValueError(f"Stage mismatch: ckpt stage={saved_stage}, "
                             f"current stage={expected_stage}")

        msg = action_model.load_state_dict(ckpt['model_state_dict'], strict=True)
        logger.info(f"Model loaded. missing={len(msg.missing_keys)}, "
                    f"unexpected={len(msg.unexpected_keys)}")

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            steps_done = ckpt['epoch'] * len(train_dataloader)
            for _ in range(steps_done):
                scheduler.step()
            logger.warning("Old checkpoint without scheduler_state_dict; "
                           "scheduler advanced manually (lr may drift slightly).")

        start_epoch = ckpt['epoch'] 
        global_step = ckpt.get('global_step',
                               start_epoch * len(train_dataloader) // args.grad_accum_steps)

        logger.info(f"Resumed at epoch={start_epoch}, global_step={global_step}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stage_name = "stage1" if args.stage1 else "stage2"
    run_save_dir = os.path.join(args.save_dir, stage_name, 'ig_' + timestamp)
    os.makedirs(run_save_dir, exist_ok=True)
    
    OmegaConf.save(config, os.path.join(run_save_dir, "config.yaml"))
    logger.info(f"Checkpoints will be saved to: {run_save_dir}")

    loss_logger = LossLogger(log_dir="log/loss")

    for epoch in range(start_epoch, args.epochs):
        model.train() 
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        start_time = time.time()

        for step, batch in enumerate(train_dataloader):
            
            with torch.amp.autocast('cuda', dtype=dtype):
                loss, info_dic = model(batch)
                loss = loss / args.grad_accum_steps

            loss.backward()

            current_step_loss = loss.item() * args.grad_accum_steps
            
            epoch_loss += loss.item() * args.grad_accum_steps
            
            # Gradient Accumulation Step
            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                loss_logger.log(epoch + 1, step + 1, global_step, current_step_loss)

                # Logging
                if global_step % 20 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time
                    
                    log_msg = (
                        f"Epoch [{epoch+1}/{args.epochs}] "
                        f"Step [{step+1}/{len(train_dataloader)}] "
                        f"Loss: {info_dic['loss_mse'] * args.grad_accum_steps:.4f} "
                        f"LR: {current_lr:.2e} "
                        f"Decay Gamma: {info_dic['mean_gamma']:.4f} "
                    )
                    
                    if args.use_next_chunk_pred:
                        log_msg += f"NextChunk: {info_dic.get('loss_next_chunk', 0.0):.4f} "
                    
                    logger.info(log_msg)

        # End of Epoch
        avg_loss = epoch_loss / len(train_dataloader)
        logger.info(f"=== Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} ===")

        # Checkpoint Saving
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            ckpt_name = f"checkpoint_epoch_{epoch+1}.pt"
            ckpt_path = os.path.join(run_save_dir, ckpt_name)
            
            save_dict = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': action_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'stage': 1 if args.stage1 else 2
            }
            
            torch.save(save_dict, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    logger.info("Training Complete.")
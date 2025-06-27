import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import math
import os
import logging
import time
import json
from datetime import datetime
from tqdm import tqdm
from dataset import MahjongGBDataset, AugmentedMahjongGBDataset
from model import CNNModel

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-8, eta_min=1e-8, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.last_epoch / self.warmup_epochs)
                    for base_lr in self.base_lrs]
        else:
            # Adjusted to avoid division by zero if max_epochs == warmup_epochs
            if self.max_epochs == self.warmup_epochs:
                return [base_lr for base_lr in self.base_lrs]
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]

def setup_logging(logdir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logdir, f'training_log_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_checkpoint(model, optimizer, scheduler, epoch, train_acc, val_acc, logdir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    checkpoint_path = os.path.join(logdir, f'checkpoint_epoch_{epoch+1}.pkl')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

if __name__ == '__main__':
    logdir = 'model/'
    data_prefix = './'

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    logger = setup_logging(logdir)

    # Load dataset
    splitRatio = 0.9
    batchSize = 128

    logger.info("Loading datasets...")
    originalDataset = MahjongGBDataset(begin=0, end=splitRatio, augment=True, data_dir_prefix=data_prefix + "data/")
    trainDataset = AugmentedMahjongGBDataset(originalDataset, augmentation_factor=1)
    validateDataset = MahjongGBDataset(begin=splitRatio, end=1, augment=False, data_dir_prefix=data_prefix + "data/")

    loader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
    vloader = DataLoader(dataset=validateDataset, batch_size=batchSize, shuffle=False)
    
    logger.info(f"Train dataset size: {len(trainDataset)}, Validation dataset size: {len(validateDataset)}")

    # Load RL model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNModel().to(device)
    
    logger.info(f"Model loaded on device: {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    scheduler = WarmupCosineScheduler(optimizer, max_epochs=num_epochs, warmup_epochs=0, warmup_start_lr=1e-6)
    
    # Define loss functions for both policy and value heads
    policy_loss_fn = F.cross_entropy
    value_loss_fn = F.mse_loss
    value_loss_weight = 0.5

    logger.info(f"Starting pre-training on {device} for {num_epochs} epochs.")

    train_history = {'epoch': [], 'train_acc': [], 'val_acc': [], 'policy_loss': [], 'value_loss': []}
    best_val_acc = 0.0
    start_time = time.time()

    for e in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f'Epoch {e+1}/{num_epochs}')
        model.train()
        _correct = 0
        total_samples = 0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0

        train_pbar = tqdm(loader, desc=f'Epoch {e+1}/{num_epochs} - Training', leave=False)
        
        for i, d in enumerate(train_pbar):
            observations, action_masks, actions, outcomes = d

            input_dict = {
                "observation": observations.to(device),
                "action_mask": action_masks.to(device)
            }

            logits, values = model(input_dict)

            pred = logits.argmax(dim=1)
            _correct += torch.eq(pred, actions.to(device)).sum().item()
            total_samples += actions.size(0)

            # Calculate separate losses for policy and value heads
            policy_loss = policy_loss_fn(logits, actions.long().to(device))
            value_loss = value_loss_fn(values.squeeze(), outcomes.to(device).float())
            
            # Combine the losses
            loss = policy_loss + value_loss_weight * value_loss

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()

            train_pbar.set_postfix({
                'P_Loss': f'{policy_loss.item():.4f}',
                'V_Loss': f'{value_loss.item():.4f}',
                'Acc': f'{_correct/total_samples:.4f}'
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_policy_loss = epoch_policy_loss / len(loader)
        avg_value_loss = epoch_value_loss / len(loader)
        train_acc = _correct / total_samples if total_samples > 0 else 0
        
        # Validation
        model.eval()
        correct_val = 0
        total_val_samples = 0
        val_policy_loss = 0.0
        val_value_loss = 0.0
        
        val_pbar = tqdm(vloader, desc=f'Epoch {e+1}/{num_epochs} - Validation', leave=False)
        
        with torch.no_grad():
            for i, d_val in enumerate(val_pbar):
                observations_val, action_masks_val, actions_val, outcomes_val = d_val
                input_dict_val = {
                    "observation": observations_val.to(device),
                    "action_mask": action_masks_val.to(device)
                }
                logits_val, values_val = model(input_dict_val)
                pred_val = logits_val.argmax(dim=1)
                correct_val += torch.eq(pred_val, actions_val.to(device)).sum().item()
                total_val_samples += actions_val.size(0)
                
                val_policy_loss += policy_loss_fn(logits_val, actions_val.long().to(device)).item()
                val_value_loss += value_loss_fn(values_val.squeeze(), outcomes_val.to(device).float()).item()

                val_pbar.set_postfix({
                    'Val_Acc': f'{correct_val/total_val_samples:.4f}' if total_val_samples > 0 else '0.0000'
                })

        val_acc = correct_val / total_val_samples if total_val_samples > 0 else 0
        avg_val_policy_loss = val_policy_loss / len(vloader)
        avg_val_value_loss = val_value_loss / len(vloader)
        epoch_time = time.time() - epoch_start_time

        train_history['epoch'].append(e+1)
        train_history['train_acc'].append(train_acc)
        train_history['val_acc'].append(val_acc)
        train_history['policy_loss'].append(avg_policy_loss)
        train_history['value_loss'].append(avg_value_loss)

        logger.info(f'Epoch {e+1}/{num_epochs} - '
                   f'Train P_Loss: {avg_policy_loss:.4f}, Train V_Loss: {avg_value_loss:.4f}, Train Acc: {train_acc:.4f}, '
                   f'Val P_Loss: {avg_val_policy_loss:.4f}, Val V_Loss: {avg_val_value_loss:.4f}, Val Acc: {val_acc:.4f}, '
                   f'Time: {epoch_time:.2f}s, LR: {scheduler.get_last_lr()[0]:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(logdir, 'best_model.pkl')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'New best model saved with validation accuracy: {val_acc:.4f}')

        if (e + 1) % 5 == 0:
            checkpoint_path = save_checkpoint(model, optimizer, scheduler, e, train_acc, val_acc, logdir)
            logger.info(f'Checkpoint saved: {checkpoint_path}')

        scheduler.step()

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")

    history_path = os.path.join(logdir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(train_history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    pretrain_model_path = os.path.join(logdir, 'pretrained_rl_cnn.pkl')
    torch.save(model.state_dict(), pretrain_model_path)
    logger.info(f"Final pre-trained model saved to {pretrain_model_path}")
    
    logger.info("="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total epochs: {num_epochs}")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Final train accuracy: {train_history['train_acc'][-1]:.4f}")
    logger.info(f"Final validation accuracy: {train_history['val_acc'][-1]:.4f}")
    logger.info(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}min)")
    logger.info("="*50)
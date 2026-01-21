import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your modules
from data.cerra_dataset import CERRADataset
from models.adaptfno_inpainting import AdaptFNOInpainting
from utils.loss import InpaintingLoss

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # 1. Load Configuration
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading config from: {args.config}")

    # 2. Setup Directories
    save_dir = config['training']['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 3. Load Datasets
    print("Initializing Training Dataset...")
    train_dataset = CERRADataset(
        gt_path=config['dataset']['train_gt'], 
        mask_path=config['dataset']['train_mask'],
        gt_var_name=config['dataset'].get('gt_var'),
        mask_var_name=config['dataset'].get('mask_var')
    )
    
    print("Initializing Validation Dataset...")
    val_dataset = CERRADataset(
        gt_path=config['dataset']['val_gt'], 
        mask_path=config['dataset']['val_mask'],
        gt_var_name=config['dataset'].get('gt_var'),
        mask_var_name=config['dataset'].get('mask_var')
    )

    # 4. Auto-Detect Resolution (Crucial for avoiding crashes)
    sample_shape = train_dataset[0]["model_input"].shape # (1, 2, H, W)
    actual_h, actual_w = sample_shape[-2], sample_shape[-1]
    print(f"--> DETECTED DATA RESOLUTION: {actual_h}x{actual_w}")
    
    # Check patch divisibility
    patch_h, patch_w = config['architecture']['patch_size']
    if actual_h % patch_h != 0 or actual_w % patch_w != 0:
        print(f"WARNING: Image size ({actual_h},{actual_w}) is not divisible by patch size ({patch_h},{patch_w})!")
        print("Model may crash. Please adjust patch_size in config.yaml.")

    # 5. Create Loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 6. Initialize Model
    print("Initializing AdaptFNO Model...")
    arch = config['architecture']
    model = AdaptFNOInpainting(
        in_channels=2,   # Sparse Wind + Mask
        out_channels=1,  # Reconstructed Wind
        img_size=(actual_h, actual_w), 
        patch_size=tuple(arch['patch_size']),
        embedding_dim=arch['embedding_dim'],
        n_layers=arch['n_layers'],
        block_size=arch['block_size'],
        dropout=arch['dropout_rate'],
        global_downsample_factor=arch['global_downsample_factor']
    ).to(device)

    # 7. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    criterion = InpaintingLoss()

    # 8. Training Loop
    best_val_loss = float('inf')
    epochs = config['training']['n_epochs']

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in loop:
            inputs = batch["model_input"].to(device)
            targets = batch["ground_truth"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs = batch["model_input"].to(device)
                targets = batch["ground_truth"].to(device)
                masks = batch["mask"].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.5f}, Val Loss = {avg_val_loss:.5f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved Best Model to {save_path}")
            
        # Regular Checkpoint
        if (epoch + 1) % config['training']['save_frequency'] == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args)
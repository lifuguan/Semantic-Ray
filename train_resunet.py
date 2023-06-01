import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from network.ops import ResUNetLight
from dataset.semantic_dataset import RandomRendererDataset, OrderRendererDataset
from network.loss import SemanticLoss
from network.metrics import IoU

import torch
import numpy as np

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, losses):
    net.eval()
    num_val_batches = len(dataloader)
    eval_results = {}

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            for loss in losses:
                # predict the mask
                masks_pred = net(image.permute(0,3,1,2))
                masks_pred = F.interpolate(
                    masks_pred, size=(240, 320), mode="bilinear", align_corners=False
                    ).permute(0,2,3,1)
                loss_results=loss({"pixel_label_nr":masks_pred, "pixel_label_gt":mask_true}, None, None)
                for k,v in loss_results.items():
                    if type(v)==torch.Tensor:
                        v=v.detach().cpu().numpy()

                    if k in eval_results:
                        eval_results[k].append(v)
                    else:
                        eval_results[k]=[v]
    for k,v in eval_results.items():
        if np.isscalar(v):
            v = np.expand_dims(v, axis=0)
        eval_results[k]=np.mean(np.concatenate(v,axis=0))
    net.train()
    return eval_results

def train_model(
        model,
        device,
        iters: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        wandb_name='ResUNet',
):
    dir_checkpoint = Path('./out/'+wandb_name)
    if 'Random'in wandb_name:
        train_set = RandomRendererDataset(is_train=True)
        val_set = OrderRendererDataset(is_train=False)
    elif 'Order'in wandb_name:
        train_set = OrderRendererDataset(is_train=True)
        val_set = OrderRendererDataset(is_train=False)
    n_train = len(train_set)
    n_val = len(val_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=64, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(
            # set the wandb project where this run will be logged
            entity="lifuguan",
            project="General-NeRF",
            name=wandb_name,
        )

    logging.info(f'''Starting training:
        iters:          {iters}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    optimizer = optim.Adam(model.parameters(),
                              lr=1e-3, weight_decay=1.0e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion = SemanticLoss({"ignore_label":20, "semantic_loss_scale": 0.25})
    evaluator = IoU({"ignore_label":20, "semantic_loss_scale": 0.25})
    # 5. Begin training
    model.train()
    global_step = 0
    iters_loss = 0
    with tqdm(total=iters, desc=f'Iter {global_step}/{iters}', unit='img') as pbar:
        while True:
            if global_step == iters: break
            for batch in train_loader: 
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images.permute(0,3,1,2))
                    masks_pred = F.interpolate(
                        masks_pred, size=(240, 320), mode="bilinear", align_corners=False
                        ).permute(0,2,3,1)
                    loss_semantic = criterion({"pixel_label_nr":masks_pred, "pixel_label_gt":true_masks},None, global_step)
                    loss = loss_semantic['loss_semantic']

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                iters_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'iter': global_step
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                if (global_step+1) % 2000 == 0:
                    histograms = {}
                    for tag, value in model.named_parameters():
                        tag = tag.replace('/', '.')
                        if not (torch.isinf(value) | torch.isnan(value)).any():
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_score = evaluate(model, val_loader, device, amp, [criterion, evaluator])
                    scheduler.step(val_score['miou'])

                    logging.info('Validation: {}'.format(val_score))
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'images': wandb.Image(images[0].cpu().numpy()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].float().cpu().numpy()),
                            'pred': wandb.Image(masks_pred[0].argmax(dim=2).float().cpu().numpy()),
                        },
                        **histograms
                    })
                    experiment.log(val_score)

                    if save_checkpoint:
                        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                        state_dict = model.state_dict()
                        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_iter{}.pth'.format(global_step)))
                        logging.info(f'Checkpoint {global_step} saved!')

                if global_step == iters: break


def get_args():
    parser = argparse.ArgumentParser(description='Train the ResUNet on images and target masks')
    parser.add_argument('--iters', type=int, default=30000, help='Number of iters')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--name', type=str, default='Order', help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=20, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # out_dim is the number of probabilities you want to get per pixel
    model = ResUNetLight(out_dim=20+1)
    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    train_model(
        model=model,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        amp=args.amp,
        wandb_name=args.name
    )




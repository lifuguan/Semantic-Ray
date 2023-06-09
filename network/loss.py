import torch
import torch.nn as nn

from network.ops import interpolate_feats


class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys = keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass


class RenderLoss(Loss):
    default_cfg = {
        'use_ray_mask': True,
        'use_dr_loss': False,
        'use_dr_fine_loss': False,
        'use_nr_fine_loss': False,
        'render_loss_scale': 1.0,
    }

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        super().__init__([f'loss_rgb'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        rgb_gt = data_pr['pixel_colors_gt']  # 1,rn,3
        rgb_nr = data_pr['pixel_colors_nr']  # 1,rn,3

        def compute_loss(rgb_pr, rgb_gt):
            loss = torch.sum((rgb_pr-rgb_gt)**2, -1)        # b,n
            if self.cfg['use_ray_mask']:
                ray_mask = data_pr['ray_mask'].float()  # 1,rn
                loss = torch.sum(loss*ray_mask, 1) / \
                    (torch.sum(ray_mask, 1)+1e-3)
            else:
                loss = torch.mean(loss, 1)
            return loss * self.cfg['render_loss_scale']

        results = {'loss_rgb_nr': compute_loss(rgb_nr, rgb_gt)}
        if self.cfg['use_dr_loss']:
            rgb_dr = data_pr['pixel_colors_dr']  # 1,rn,3
            results['loss_rgb_dr'] = compute_loss(rgb_dr, rgb_gt)
        if self.cfg['use_dr_fine_loss']:
            results['loss_rgb_dr_fine'] = compute_loss(
                data_pr['pixel_colors_dr_fine'], rgb_gt)
        if self.cfg['use_nr_fine_loss']:
            results['loss_rgb_nr_fine'] = compute_loss(
                data_pr['pixel_colors_nr_fine'], rgb_gt)
        return results


class DepthLoss(Loss):
    default_cfg = {
        'depth_correct_thresh': 0.02,
        'depth_loss_type': 'l2',
        'depth_loss_l1_beta': 0.05,
    }

    def __init__(self, cfg):
        super().__init__(['loss_depth'])
        self.cfg = {**self.default_cfg, **cfg}
        if self.cfg['depth_loss_type'] == 'smooth_l1':
            self.loss_op = nn.SmoothL1Loss(
                reduction='none', beta=self.cfg['depth_loss_l1_beta'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'true_depth' not in data_gt['ref_imgs_info']:
            return {'loss_depth': torch.zeros([1], dtype=torch.float32, device=data_pr['pixel_colors_nr'].device)}
        coords = data_pr['depth_coords']  # rfn,pn,2
        depth_pr = data_pr['depth_mean']  # rfn,pn
        depth_maps = data_gt['ref_imgs_info']['true_depth']  # rfn,1,h,w
        rfn, _, h, w = depth_maps.shape
        depth_gt = interpolate_feats(
            depth_maps, coords, h, w, padding_mode='border', align_corners=True)[..., 0]   # rfn,pn

        # transform to inverse depth coordinate
        depth_range = data_gt['ref_imgs_info']['depth_range']  # rfn,2
        near, far = -1/depth_range[:, 0:1], -1/depth_range[:, 1:2]  # rfn,1

        def process(depth):
            depth = torch.clamp(depth, min=1e-5)
            depth = -1 / depth
            depth = (depth - near) / (far - near)
            depth = torch.clamp(depth, min=0, max=1.0)
            return depth
        depth_gt = process(depth_gt)

        # compute loss
        def compute_loss(depth_pr):
            if self.cfg['depth_loss_type'] == 'l2':
                loss = (depth_gt - depth_pr)**2
            elif self.cfg['depth_loss_type'] == 'smooth_l1':
                loss = self.loss_op(depth_gt, depth_pr)

            if data_gt['scene_name'].startswith('gso'):
                # rfn,1,h,w
                depth_maps_noise = data_gt['ref_imgs_info']['depth']
                depth_aug = interpolate_feats(
                    depth_maps_noise, coords, h, w, padding_mode='border', align_corners=True)[..., 0]  # rfn,pn
                depth_aug = process(depth_aug)
                mask = (torch.abs(depth_aug-depth_gt) <
                        self.cfg['depth_correct_thresh']).float()
                loss = torch.sum(loss * mask, 1) / (torch.sum(mask, 1) + 1e-4)
            else:
                loss = torch.mean(loss, 1)
            return loss

        outputs = {'loss_depth': compute_loss(depth_pr)}
        if 'depth_mean_fine' in data_pr:
            outputs['loss_depth_fine'] = compute_loss(
                data_pr['depth_mean_fine'])
        return outputs


class SemanticLoss(Loss):
    def __init__(self, cfg):
        super().__init__(['loss_semantic'])
        self.cfg = cfg
        self.ignore_label = cfg['ignore_label']

    def __call__(self, data_pr, data_gt, step, **kwargs):
        num_classes = data_pr['pixel_label_nr'].shape[-1]
        def compute_loss(label_pr, label_gt):
            label_pr = label_pr.reshape(-1, num_classes)
            label_gt = label_gt.reshape(-1).long()
            valid_mask = (label_gt != self.ignore_label)
            label_pr = label_pr[valid_mask]
            label_gt = label_gt[valid_mask]
            return nn.functional.cross_entropy(label_pr, label_gt, reduction='mean').unsqueeze(0)
        
        pixel_label_gt = data_pr['pixel_label_gt']
        pixel_label_nr = data_pr['pixel_label_nr']
        coarse_loss = compute_loss(pixel_label_nr, pixel_label_gt)
        
        if 'pixel_label_gt_fine' in data_pr:
            pixel_label_gt_fine = data_pr['pixel_label_gt_fine']
            pixel_label_nr_fine = data_pr['pixel_label_nr_fine']
            fine_loss = compute_loss(pixel_label_nr_fine, pixel_label_gt_fine)
        else:
            fine_loss = torch.zeros_like(coarse_loss)
        
        loss = (coarse_loss + fine_loss) * self.cfg['semantic_loss_scale']
        
        if 'pred_labels' in data_pr:
            ref_labels_pr = data_pr['pred_labels'].permute(0, 2, 3, 1)
            ref_labels_gt = data_gt['ref_imgs_info']['labels'].permute(0, 2, 3, 1)
            ref_loss = compute_loss(ref_labels_pr, ref_labels_gt)
            loss += ref_loss * self.cfg['semantic_loss_scale']
        return {'loss_semantic': loss}


name2loss = {
    'render': RenderLoss,
    'depth': DepthLoss,
    'semantic': SemanticLoss,
}

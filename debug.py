# %%

import numpy as np
import matplotlib.pyplot as plt
plt.clf()
plt.figure(dpi=600)
fig = plt.figure()
attn_vis = attn[0].to('cpu').detach().numpy().sum(axis=2)
plt.imshow(attn_vis)
plt.gca().set_aspect(0.2)
plt.colorbar()
plt.savefig('out/view_attn.pdf', bbox_inches='tight')

# %%
import torch
import numpy
from gnt.feature_network import ResUNet
feature_net = ResUNet(
            coarse_out_ch=32,
            fine_out_ch=32,
            single_net=True,
        ).to("cuda:1")
checkpoint = torch.load('./out/gnt_full/model_720000.pth')
feature_net.load_state_dict(checkpoint['feature_net'])


img = torch.from_numpy(numpy.load('out/gt_img.npy')).to("cuda:1")
out = feature_net(img.permute(2,0,1).unsqueeze(0))[2]
print(out.shape)
numpy.save("out/gt_feat.npy", out.to('cpu').detach().numpy())

# %%
import torch
import numpy as np
feat_2d = torch.from_numpy(np.load("out/feat_2d.npy")).to("cuda:1")
roi = feat_2d[::30, ::30,:]

# out = torch.from_numpy(np.load("out/gt_feat.npy")).to("cuda:1")
# print(feat_2d.shape, roi.shape, out.shape)


# %%

keys = out #/ out.norm(dim=1, keepdim=True)
queries = roi #/ roi.norm(dim=1, keepdim=True)
h, w, c,_ = queries.shape
queries = queries.reshape(h*w, c, -1).unsqueeze(-1)
print(h,w,queries.shape, keys.shape)

# %%

attn = torch.nn.functional.conv2d(keys, queries, stride=1)

attn = attn.reshape(h, w, 192, 252).to('cpu').detach().numpy()

print(attn.shape)

# %%

attn = attn > 0

# %%
import matplotlib.pyplot as plt
fig,a =  plt.subplots(h,w)
for i, per_img_masks in enumerate(attn):
    for j, per_mask in enumerate(per_img_masks):
        norm_mask = per_mask / np.linalg.norm(per_mask)
        a[i][j].set_axis_off()
        a[i][j].imshow(norm_mask)
plt.axis("off")
plt.savefig('out/feat_attn.pdf', bbox_inches='tight')

# %%
import torch
import numpy as np
feat_2d = torch.from_numpy(np.load("out/feat_2d.npy")).to("cuda:1")
roi = feat_2d[250:252, 190:192, 32:]

keys = out / out.norm(dim=0, keepdim=True)
queries = roi / roi.norm(dim=1, keepdim=True)
attn = queries @ keys.reshape(keys.shape[0], -1)
h, w = attn.shape[:2]
attn = attn.reshape(h, w, 192, 252).to('cpu').detach().numpy()


# %%
import torch

# 定义 query 和 key tensor
query = torch.randn(1, 13, 17, 256)
key = torch.randn(1, 100, 200, 256)

# 将 query 和 key 分别转换为形状为 (1, 256, 13, 17) 和 (1, 256, 100, 200) 的 tensor
query = query.permute(0, 3, 1, 2)
key = key.permute(0, 3, 1, 2)

# 使用 2D 卷积计算 correlation map
corr_map = torch.nn.functional.conv2d(key, query, stride=1)

# 转换 correlation map 的形状为 (1, 13, 17, 100, 200)
corr_map = corr_map.permute(0, 2, 3, 1)


# %%
norm_mask = attn[6][6] # / np.linalg.norm(attn[1][5])
np.min(norm_mask)
# norm_mask = norm_mask > 0.9
plt.imshow(norm_mask)
plt.colorbar()

# %%
keys.unique()

# %%
np.unique(attn)

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
for pose in all_poses:
    # 姿态的位置信息
    position = pose[:3, 3]
    x, y, z = position

    # 姿态的旋转信息
    rotation = pose[:3, :3]

    # 绘制位置
    ax.scatter(x, y, z, c='r', marker='o')

    # 绘制方向箭头
    arrow_length = 0.1
    ax.quiver(x, y, z, rotation[0, 0], rotation[1, 0], rotation[2, 0], length=arrow_length, color='r')
    ax.quiver(x, y, z, rotation[0, 1], rotation[1, 1], rotation[2, 1], length=arrow_length, color='g')
    ax.quiver(x, y, z, rotation[0, 2], rotation[1, 2], rotation[2, 2], length=arrow_length, color='b')


# # 设置坐标轴范围
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])
# ax.set_zlim([zmin, zmax])

# 显示图形
plt.show()



# %%
import torch
params = torch.load("out/resunet/checkpoint_epoch8.pth")

resunet = {'network_state_dict':{}}
for key in params.keys():
    resunet['network_state_dict']['semantic_branch.'+key] = params[key]
torch.save(resunet, "out/resunet/resunet_ep8.pth")

# %%
param.keys()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from dataset.semantic_dataset import StatisticRendererDataset
from tqdm import tqdm

import torch
import numpy as np

train_set = StatisticRendererDataset(is_train=True)
# 3. Create data loaders
loader_args = dict(batch_size=1, num_workers=8, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
global_step = 0
iters = len(train_loader)
scene_statistic = []
with tqdm(total=iters, desc=f'Iter {global_step}/{iters}', unit='img') as pbar:
    for batch in train_loader: 
        scene_statistic.append(batch)
        pbar.update(1)


np.save("out/scene_statistic.npy", np.array(scene_statistic))




# %%

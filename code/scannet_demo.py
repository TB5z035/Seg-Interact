# Necessary for advanced config parsing with hydra and omegaconf
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import torch
from .sp_visualization import show
from .dataset.scannet import CLASS_NAMES, CLASS_COLORS
from .dataset.scannet import ScanNet_NUM_CLASSES as NUM_CLASSES
from .dataset.sp_transforms import *
from .utils.args import get_args
from .dataset import DATASETS

from grid_graph import edge_list_to_forward_star

cfg, _ = get_args()
cfg_dm = cfg.datamodule

dataset = DATASETS[cfg.datamodule['sp_base']](**(cfg.train_dataset['args'] | {'sp_cfg': cfg_dm}))

nag = dataset[0]
num_points = nag.num_points

##### Graphing #####

point_coords = nag[0].pos
sp1_coords = nag[1].pos
point_edge_idx = nag[0].edge_index
sp1_edge_idx = nag[1].edge_index

point_edge_source = point_coords[point_edge_idx[0]].permute(1, 0).unsqueeze(dim=2)
point_edge_target = point_coords[point_edge_idx[1]].permute(1, 0).unsqueeze(dim=2)
point_edge = torch.concat((point_edge_source, point_edge_target), dim=2)
sp1_edge_source = sp1_coords[sp1_edge_idx[0]].permute(1, 0).unsqueeze(dim=2)
sp1_edge_target = sp1_coords[sp1_edge_idx[1]].permute(1, 0).unsqueeze(dim=2)
sp1_edge = torch.concat((sp1_edge_source, sp1_edge_target), dim=2)
edges = torch.concat((point_edge, sp1_edge), dim=1)

Xn = torch.concat((point_coords[:, 0], sp1_coords[:, 0])).numpy()  # x-coordinates of nodes
Yn = torch.concat((point_coords[:, 1], sp1_coords[:, 1])).numpy()  # y-coordinates of nodes
Zn = torch.concat((point_coords[:, 2], sp1_coords[:, 2])).numpy()  # z-coordinates of nodes
Xe = []
Ye = []
Ze = []

N = edges.shape[1]
for i in range(N):
    Xe += [edges[0][i][0], edges[0][i][1], None]
    Ye += [edges[1][i][0], edges[1][i][1], None]
    Ze += [edges[2][i][0], edges[2][i][1], None]

# Node Colors by Superpoint Grouping
super_index_10 = nag.get_super_index(1, 0)
super_index_21 = nag.get_super_index(2, 1)
point_colors = (super_index_10 / len(torch.unique(super_index_10)))
sp1_colors = (super_index_21 / len(torch.unique(super_index_21)))
node_colors = torch.cat((point_colors, sp1_colors)).numpy()

# Edge Colors by Edge Weighting
reg = [0.01, 0.1]

_, _, point_reindex = edge_list_to_forward_star(nag[0].num_nodes, nag[0].edge_index.T.contiguous().cpu().numpy())
point_edge_weights = nag[0].edge_attr.cpu().numpy()[point_reindex] * reg[0]
point_edge_colors = point_edge_weights/np.max(point_edge_weights)

_, _, sp1_reindex = edge_list_to_forward_star(nag[1].num_nodes, nag[1].edge_index.T.contiguous().cpu().numpy())
sp1_edge_weights = nag[1].edge_attr.cpu().numpy()[sp1_reindex] * reg[1]
sp1_edge_weights = sp1_edge_weights[:, 6].transpose()
sp1_edge_weights = sp1_edge_weights/np.max(sp1_edge_weights)
edge_weights = np.concatenate((point_edge_weights, sp1_edge_weights), axis=0)
edge_colors = edge_weights/np.max(edge_weights)
edge_colors = np.expand_dims(edge_colors, axis=0)
edge_colors = np.ravel(np.tile(edge_colors, (3,1)).transpose())

print(len(Xe), len(Ye), len(Ze))
print(Xn.shape, Yn.shape, Zn.shape)
print(node_colors.shape, edge_colors.shape)

# Configs and Plot
edge_trace = go.Scatter3d(x=Xe,
                          y=Ye,
                          z=Ze,
                          mode='lines',
                          line=dict(color=edge_colors, width=2, colorscale='viridis'),
                          hoverinfo='none')

node_trace = go.Scatter3d(x=Xn,
                          y=Yn,
                          z=Zn,
                          mode='markers',
                          name='nodes',
                          marker=dict(symbol='circle',
                                      size=4,
                                      color=node_colors,
                                      colorscale='sunset',
                                      line=dict(color='rgb(50,50,50)', width=0.5)),
                          hoverinfo='text')

axis = dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')

layout = go.Layout(
    title="Superpoint Graph",
    width=2000,
    height=2000,
    showlegend=True,
    scene=dict(
        xaxis=dict(axis),
        yaxis=dict(axis),
        zaxis=dict(axis),
    ),
    margin=dict(t=100),
    hovermode='closest',
    annotations=[
        dict(showarrow=False,
             text='temp',
             xref='paper',
             yref='paper',
             x=0,
             y=0.1,
             xanchor='left',
             yanchor='bottom',
             font=dict(size=14))
    ],
)

data = [node_trace, edge_trace]
fig = go.Figure(data=data, layout=layout)
fig.show()


'''
mode_num = []
percentage = []

for i in range(sup_max):
    inds = torch.where(sup_inds == i)[0]
    sup_labels = labels[inds]
    main_num = torch.mode(sup_labels)[0]
    main_freq = torch.bincount(sup_labels)[main_num]
    percent = main_freq / len(sup_labels)

    mode_num.append(main_num)
    percentage.append(percent)

mode_num = torch.tensor(mode_num)
percentage = torch.tensor(percentage)

print(mode_num)
print(percentage)
print(f'average contain mode_num percentage: {torch.sum(percentage)/len(percentage)}')
print(f'average pure mode_num percentage: {len(percentage[percentage == 1.])/len(percentage)}')


show(nag, class_names=CLASS_NAMES, ignore=NUM_CLASSES, class_colors=CLASS_COLORS, max_points=200000)
'''

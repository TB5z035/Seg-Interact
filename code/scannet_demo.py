# Necessary for advanced config parsing with hydra and omegaconf
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import torch
from tqdm import tqdm

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

lv0 = []
lv1 = []
lv2 = []
lv3 = []
print(dataset[944])
exit()

for i in tqdm(range(len(dataset))):
    nag = dataset[i]
    num_points = nag.num_points
    lv0.append(num_points[0])
    # lv1.append(torch.max(nag.get_sub_size(1,0)))
    # lv2.append(torch.max(nag.get_sub_size(2,0)))
    # lv3.append(torch.max(nag.get_sub_size(3,0)))

print(np.min(lv0))
print(np.where(lv0 == np.min(lv0)))

# print(torch.mean(torch.tensor(lv1), dtype=float), torch.mean(torch.tensor(lv2), dtype=float), torch.mean(torch.tensor(lv3), dtype=float))
# print(torch.min(torch.tensor(lv1)), torch.min(torch.tensor(lv2)), torch.min(torch.tensor(lv3)))
# print(torch.max(torch.tensor(lv1)), torch.max(torch.tensor(lv2)), torch.max(torch.tensor(lv3)))
exit()

##### Graphing #####

point_coords = nag[0].pos
sp2_coords = nag[2].pos
sp2_edge_source, sp2_edge_target = nag[2].edge_index[0], nag[2].edge_index[1]
super_index_20 = nag.get_super_index(2, 0)
color_map = {'source': 'rgb(255,0,0)', 'neighbor': 'rgb(255,255,0)', 'other': 'rgb(0,0,255)'}

source_sp2_index = 0
neighbor_sp2_index = sp2_edge_target[torch.where(sp2_edge_source == source_sp2_index)[0]]

other_sp2_index = torch.unique(super_index_20)
unwanted_index = torch.cat((neighbor_sp2_index, torch.tensor([source_sp2_index])))
other_sp2_index[unwanted_index] = -1
other_sp2_index = other_sp2_index[other_sp2_index != -1]

Xn_p = point_coords[:, 0].numpy()  # x-coordinates of nodes
Yn_p = point_coords[:, 1].numpy()  # y-coordinates of nodes
Zn_p = point_coords[:, 2].numpy()  # z-coordinates of nodes

node_colors = np.zeros(len(super_index_20))  # [0]*len(torch.unique(super_index_20))
source_nodes = torch.where(super_index_20 == source_sp2_index)[0].numpy()
neighbor_nodes = []
other_nodes = []
for i in range(len(neighbor_sp2_index)):
    neighbor_nodes.append(torch.where(super_index_20 == neighbor_sp2_index[i])[0])
for j in range(len(other_sp2_index)):
    other_nodes.append(torch.where(super_index_20 == other_sp2_index[j])[0])
neighbor_nodes = torch.cat(neighbor_nodes).numpy()
other_nodes = torch.cat(other_nodes).numpy()
node_colors[source_nodes] = 0  # color_map['source']
node_colors[neighbor_nodes] = 1  # color_map['neighbor']
node_colors[other_nodes] = 2  # color_map['other']
'''
point_edge_source = point_coords[point_edge_idx[0]].permute(1, 0).unsqueeze(dim=2)
point_edge_target = point_coords[point_edge_idx[1]].permute(1, 0).unsqueeze(dim=2)
point_edge = torch.concat((point_edge_source, point_edge_target), dim=2)
sp1_edge_source = sp1_coords[sp1_edge_idx[0]].permute(1, 0).unsqueeze(dim=2)
sp1_edge_target = sp1_coords[sp1_edge_idx[1]].permute(1, 0).unsqueeze(dim=2)
sp1_edge = torch.concat((sp1_edge_source, sp1_edge_target), dim=2)
# edges = torch.concat((point_edge, sp1_edge), dim=1)

Xn_p, Xn_sp = point_coords[:, 0].numpy(), sp1_coords[:, 0].numpy()  # x-coordinates of nodes
Yn_p, Yn_sp = point_coords[:, 1].numpy(), sp1_coords[:, 1].numpy()  # y-coordinates of nodes
Zn_p, Zn_sp = point_coords[:, 2].numpy(), sp1_coords[:, 2].numpy()  # z-coordinates of nodes
Xe_p = []
Ye_p = []
Ze_p = []
Xe_sp = []
Ye_sp = []
Ze_sp = []

N_p = point_edge.shape[1]
N_sp = sp1_edge.shape[1]
for i in range(N_p):
    Xe_p += [point_edge[0][i][0], point_edge[0][i][1], None]
    Ye_p += [point_edge[1][i][0], point_edge[1][i][1], None]
    Ze_p += [point_edge[2][i][0], point_edge[2][i][1], None]
for i in range(N_sp):
    Xe_sp += [sp1_edge[0][i][0], sp1_edge[0][i][1], None]
    Ye_sp += [sp1_edge[1][i][0], sp1_edge[1][i][1], None]
    Ze_sp += [sp1_edge[2][i][0], sp1_edge[2][i][1], None]

# Node Colors by Superpoint Grouping
super_index_10 = nag.get_super_index(1, 0)
super_index_21 = nag.get_super_index(2, 1)
point_colors = (super_index_10 / len(torch.unique(super_index_10))).numpy()
sp1_colors = (super_index_21 / len(torch.unique(super_index_21))).numpy()

# Edge Colors by Edge Weighting
reg = [0.01, 0.1]

_, _, point_reindex = edge_list_to_forward_star(nag[0].num_nodes, nag[0].edge_index.T.contiguous().cpu().numpy())
point_edge_weights = nag[0].edge_attr.cpu().numpy()[point_reindex] * reg[0]
point_edge_weights = point_edge_weights / np.max(point_edge_weights)
point_edge_colors = np.expand_dims(point_edge_weights, axis=0)
point_edge_colors = np.ravel(np.tile(point_edge_colors, (3, 1)).transpose())

_, _, sp1_reindex = edge_list_to_forward_star(nag[1].num_nodes, nag[1].edge_index.T.contiguous().cpu().numpy())
sp1_edge_weights = nag[1].edge_attr.cpu().numpy()[sp1_reindex] * reg[1]
sp1_edge_weights = sp1_edge_weights[:, 5].transpose()
sp1_edge_weights = sp1_edge_weights / np.max(sp1_edge_weights)
sp1_edge_colors = np.expand_dims(sp1_edge_weights, axis=0)
sp1_edge_colors = np.ravel(np.tile(sp1_edge_colors, (3, 1)).transpose())

# edge_weights = np.concatenate((point_edge_weights, sp1_edge_weights), axis=0)
# edge_colors = edge_weights / np.max(edge_weights)
# edge_colors = np.expand_dims(edge_colors, axis=0)
# edge_colors = np.ravel(np.tile(edge_colors, (3, 1)).transpose())
'''

# Configs and Plot
'''
point_edge_trace = go.Scatter3d(x=Xe_p,
                          y=Ye_p,
                          z=Ze_p,
                          mode='lines',
                          name='point_edges',
                          line=dict(color=point_edge_colors, width=2, colorscale='viridis'),
                          hoverinfo='none')

sp1_edge_trace = go.Scatter3d(x=Xe_sp,
                          y=Ye_sp,
                          z=Ze_sp,
                          mode='lines',
                          name='sp1_edges',
                          line=dict(color=sp1_edge_colors, width=2, colorscale='viridis'),
                          hoverinfo='none')
'''

point_node_trace = go.Scatter3d(x=Xn_p,
                                y=Yn_p,
                                z=Zn_p,
                                mode='markers',
                                name='point_nodes',
                                marker=dict(symbol='circle',
                                            size=4,
                                            color=node_colors,
                                            line=dict(color='rgb(50,50,50)', width=0.5)),
                                hoverinfo='text')

# sp1_node_trace = go.Scatter3d(x=Xn_sp,
#                           y=Yn_sp,
#                           z=Zn_sp,
#                           mode='markers',
#                           name='sp1_nodes',
#                           marker=dict(symbol='circle',
#                                       size=4,
#                                       color='rgb(255,255,0)',
#                                       line=dict(color='rgb(50,50,50)', width=0.5)),
#                           hoverinfo='text')

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

# data = [point_node_trace, point_edge_trace, sp1_node_trace, sp1_edge_trace]
data = [point_node_trace]
# data = [point_node_trace, sp1_node_trace]
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

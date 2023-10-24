import numpy as np
import os
import os.path as osp
import re
import torch
import torch.nn as nn
from itertools import repeat
import argparse
import os.path as osp
import yaml
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


import plotly.graph_objs as go

N = 5

Xn = [k for k in range(N)]  # x-coordinates of nodes
Yn = [k for k in range(N)]  # y-coordinates
Zn = [k for k in range(N)]  # z-coordinates
Xe = []
Ye = []
Ze = []

#red green blue
group = [['rgb(150,0,0)','rgb(0,0,150)'],
         'rgb(255,0,0)','rgb(255,0,0)',
         'rgb(0,255,0)','rgb(0,255,0)',
         'rgb(255,0,0)','rgb(255,0,0)']
temp1 = [5,5,5,
         10,10,10,
         20,20,20,
         30,30,30]

for i in range(N):
    if i != (N-1):
        Xe += [i,i+1, None]
        Ye += [i,i+1, None]
        Ze += [i,i+1, None]

trace1 = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', line=dict(color=temp1, width=5, colorscale='viridis'), hoverinfo='none')

trace2 = go.Scatter3d(
    x=Xn,
    y=Yn,
    z=Zn,
    mode='markers',
    name='actors',
    marker=dict(symbol='circle', size=6, color=Zn, colorscale='Viridis', line=dict(color='rgb(50,50,50)',
                                                                                      width=0.5)),
    #    text=labels,
    hoverinfo='text')

axis = dict(showbackground=True, showline=True, zeroline=True, showgrid=True, showticklabels=True, title='')

layout = go.Layout(
    title="Testing Graph",
    width=1000,
    height=1000,
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
             text=None,
             xref='paper',
             yref='paper',
             x=0,
             y=0.1,
             xanchor='left',
             yanchor='bottom',
             font=dict(size=14))
    ],
)

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
fig.show()


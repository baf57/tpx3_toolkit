'''
Contains all of the output viewing functions for use with the TimePix3 camera.
'''

import copy
from tpx3_toolkit.core import Beam
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib import colors as c
import matplotlib.cm as cm
import numpy as np

def plot_hits(pix:np.ndarray,colorMap:str='viridis',fig:Figure=None) -> Figure: #type:ignore

    if fig is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_axes([0,0,1,1])
    else:
        ax = fig.gca()

    make_hits_axes(pix,ax,colorMap)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig

def make_hits_axes(pix:np.ndarray,ax:Axes,colorMap:str='viridis') -> None:

    CCD = np.zeros((256,256))
    indices = (pix[1,:].astype('int'),pix[0,:].astype('int'))
    np.add.at(CCD,indices,1) # adds 1 to the CCD value at each hit's (x,y)

    np.add.at(CCD,(150,50),500)

    cmap = copy.copy(cm.get_cmap(colorMap))
    if hasattr(cmap,'colors'):
        cmap.set_bad(cmap.colors[0]) #type:ignore
    else:
        cmap.set_bad(cmap(0))

    cutoff = np.max(CCD) * 0.2

    ax.imshow(CCD,origin='lower',aspect='auto',extent=[0,256,0,256],\
        vmax=cutoff,interpolation='none',cmap=cmap) #type:ignore

def draw_beam_box(ax:Axes,beams:list[Beam],boxColors:list[str]=[]) \
    -> list[LineCollection]:
    if len(boxColors) == 0:
        cmap = cm.get_cmap('Set1')
        colors = [cmap.colors[i%len(cmap.colors)] for i in range(len(beams))] #type: ignore
    elif len(boxColors) == 1:
        colors = boxColors * len(beams)
    elif len(boxColors) == len(beams):
        colors = boxColors
    else:
        "Please give either no color list, 1 color in the list, or color list equal in size to the beams list."
        return [LineCollection([])]

    out = list()
    for beam,col in zip(beams,colors):
        out.append(ax.hlines([beam.bottom,beam.top],beam.left,beam.right,col)) #type: ignore
        out.append(ax.vlines([beam.left,beam.right],beam.bottom,beam.top,col)) #type: ignore

    return out

def plot_coincidences(coincidences:np.ndarray,colorMap:str='',\
                      fig:Figure=None) -> Figure: #type: ignore

    if colorMap == '':
        # default red color map to look like a laser idk
        colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
        colorMap = c.LinearSegmentedColormap.from_list(
                "Custom", colors, N=20) #type:ignore

    if fig is None:
        fig = plt.figure(figsize=(12,6))
        ax_signal = fig.add_subplot(121)
        ax_idler = fig.add_subplot(122)
    else:
        [ax_signal,ax_idler] = fig.axes

    make_coincidences_axis(coincidences[1,:,:],ax_signal,colorMap)
    make_coincidences_axis(coincidences[0,:,:],ax_idler,colorMap)

    ax_signal.set_title("Signal")
    ax_idler.set_title("Idler")

    return fig

def make_coincidences_axis(pix:np.ndarray,ax:Axes,\
                           colorMap:str='viridis') -> None:

    xmin = np.min(pix[0,:])
    xmax = np.max(pix[0,:])
    ymin = np.min(pix[1,:])
    ymax = np.max(pix[1,:])

    xrange = int(xmax - xmin)
    yrange = int(ymax - ymin)

    view = np.zeros((xrange+1,yrange+1))

    indices = ((pix[0,:] - xmin).astype('int'),(pix[1,:]-ymin).astype('int'))
    np.add.at(view,indices,1) # adds 1 to the view value at each hit's (x,y)

    if type(colorMap) is str:
        cmap = copy.copy(cm.get_cmap(colorMap))
    else:
        cmap = copy.copy(colorMap)
    if hasattr(cmap,'colors'):
        cmap.set_bad(cmap.colors[0]) #type: ignore
    else:
        cmap.set_bad(cmap(0)) #type: ignore

    ax.imshow(view,origin='upper',aspect='auto',extent=[0,xrange,0,yrange],\
        interpolation='none',cmap=cmap)

def plot_correlations(coincidences:np.ndarray,colorMap:str="gray", \
                      fig:Figure=None) -> Figure: #type:ignore
    if fig is None:
        fig = plt.figure(figsize=(12,6))
        ax_x = fig.add_subplot(121)
        ax_y = fig.add_subplot(122)
    else:
        [ax_x,ax_y] = fig.axes

    make_coincidences_axis(coincidences[:,1,:],ax_x,colorMap)
    make_coincidences_axis(coincidences[:,0,:],ax_y,colorMap)

    ax_x.set_ylabel("Signal")
    ax_x.set_xlabel("Idler")
    ax_x.set_title("X")
    ax_y.set_ylabel("Signal")
    ax_y.set_xlabel("Idler")
    ax_y.set_title("Y")

    return fig


def plot_histogram(coincidences:np.ndarray, min_bin=-200, max_bin=200,\
                   fig:Figure=None) -> Figure: #type: ignore
    if fig is None:
        fig = plt.figure(figsize=(4,8))
        ax = fig.add_axes([0,0,1,1])
    else:
        ax = fig.gca()
        ax.clear()

    ax.set_xlabel("dt [ns]")
    ax.set_ylabel("Count")

    bins = np.linspace(min_bin,max_bin,100)
    # get difference between signal and idler arrival times
    dt = coincidences[1,2,:] - coincidences[0,2,:]

    ax.hist(dt,bins,color='r')
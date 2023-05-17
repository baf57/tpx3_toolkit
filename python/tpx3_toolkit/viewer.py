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
    print(pix[0,345:])
    print(pix[0,888:])
    indices = (pix[1,:].astype('int'),pix[0,:].astype('int'))
    np.add.at(CCD,indices,1) # adds 1 to the CCD value at each hit's (x,y)

    np.add.at(CCD,(150,50),500)

    cmap = copy.copy(cm.get_cmap(colorMap))
    if hasattr(cmap,'colors'):
        cmap.set_bad(cmap.colors[0]) #type:ignore
    else:
        cmap.set_bad(cmap(0))

    cutoff = np.max(CCD) * 0.2

    ax.imshow(CCD,origin='upper',aspect='auto',extent=[0,256,0,256],\
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
        ax_signal = fig.add_subplot(122)
        ax_idler = fig.add_subplot(121)
    else:
        [ax_idler,ax_signal] = fig.axes

    _make_coincidences_axis(coincidences[0,:,:],ax_idler,colorMap)
    _make_coincidences_axis(coincidences[1,:,:],ax_signal,colorMap)

    ax_signal.set_title("Signal")
    ax_idler.set_title("Idler")

    return fig

def plot_correlations(coincidences:np.ndarray,colorMap:str="gray", \
                      fig:Figure=None) -> Figure: #type:ignore
    if fig is None:
        fig = plt.figure(figsize=(12,6))
        ax_x = fig.add_subplot(121)
        ax_y = fig.add_subplot(122)
    else:
        [ax_x,ax_y] = fig.axes

    _make_coincidences_axis(coincidences[:,0,:],ax_x,colorMap)
    _make_coincidences_axis(coincidences[:,1,:],ax_y,colorMap)

    ax_x.set_ylabel("Signal")
    ax_x.set_xlabel("Idler")
    ax_x.set_title("X")
    ax_y.set_ylabel("Signal")
    ax_y.set_xlabel("Idler")
    ax_y.set_title("Y")

    return fig


def plot_histogram(coincidences:np.ndarray, min_bin=-200, max_bin=200,\
                   fig:Figure=None) -> Figure:
    if fig is None:
        fig = plt.figure(figsize=(4,8))
        ax = fig.add_axes([0,0,1,1])
    else:
        ax = fig.gca()

    ax.set_xlabel("dt [ns]")
    ax.set_ylabel("Count")

    bins = np.linspace(min_bin,max_bin,100)
    # get difference between signal and idler arrival times
    dt = coincidences[1,2,:] - coincidences[0,2,:]

    ax.hist(dt,bins,color='r')

    return fig

def plot_coincidence_trace(pix:np.ndarray, loc:int, orientation:str,
                           min_loc:int=0, max_loc:int=256, ax:Axes=None) \
                            -> tuple[Figure, np.ndarray, np.ndarray]:
    # pix is a coincidences matrix which has already been reduces to 
    # 2D (i.e. 1 beam x-y info, only x info for both beams, etc.)
    if ax is None:
        fig = plt.figure(figsize=(6,8))
        ax = fig.add_axes([0,0,1,1])
    else:
        fig = ax.get_figure()

    (view,x,y) = _make_view(pix)

    ax.set_xlabel(f'Index')
    ax.set_ylabel(f'Count')
    ax.set_xlim(min_loc, max_loc)

    if orientation == 'x':
        data = view[:,loc]
    else:
        data = view[loc,:]

    ax.bar(np.arange(data.size)+0.5,data,color='gray')

    return (fig,data,view)

def plot_coincidence_xy(correlations:np.ndarray, sign:int=1, fig:Figure=None)\
                                                   -> tuple[Figure,np.ndarray]:
    if fig is None:
        fig = plt.figure(figsize=(4,8))
        ax = fig.add_axes([0,0,1,1])
    else:
        ax = fig.gca()

    data = correlations[0,:,:] + (sign/np.abs(sign)) * correlations[1,:,:]
    view = _make_coincidences_axis(data,ax)

    ax.set_xlabel(r'$x_{idl} + x_{sig}$')
    ax.set_ylabel(r'$y_{idl} + y_{sig}$')

    return (fig,view)

def _make_coincidences_axis(pix:np.ndarray,ax:Axes,\
                           colorMap:str='viridis') -> None:
    (view,xrange,yrange) = _make_view(pix)

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

    return view

def _make_view(pix:np.ndarray):
    xmin = np.min(pix[0,:])
    xmax = np.max(pix[0,:])
    ymin = np.min(pix[1,:])
    ymax = np.max(pix[1,:])

    xrange = int(xmax - xmin)
    yrange = int(ymax - ymin)

    view = np.zeros((yrange+1,xrange+1))

    indices = ((pix[1,:] - ymin).astype('int'),(pix[0,:]-xmin).astype('int'))
    np.add.at(view,indices,1) # adds 1 to the view value at each hit's (x,y)

    return (view,xrange,yrange)
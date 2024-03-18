'''
Contains all of the output viewing functions for use with the TimePix3 camera.
'''

import copy
from tpx3_toolkit.core import Beam, xp, asnumpy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib import colors as c
from scipy import signal
import matplotlib.cm as cm
import numpy as np

try: # optional CuPyx import to circumvent issue with `.at()` ufunc
    from cupyx import scatter_add
    add_at = scatter_add
except:
    add_at = np.add.at

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

    CCD = xp.zeros((256,256))
    indices = (pix[1,:].astype(int),pix[0,:].astype(int))
    xp.add.at(CCD,indices,1) # adds 1 to the CCD value at each hit's (x,y)

    cmap = copy.copy(cm.get_cmap(colorMap))
    if hasattr(cmap,'colors'):
        cmap.set_bad(cmap.colors[0]) #type:ignore
    else:
        cmap.set_bad(cmap(0))

    cutoff = np.max(CCD) * 0.2

    ax.imshow(asnumpy(CCD),origin='lower',aspect='auto',extent=[0,256,0,256],\
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

    idl = _make_coincidences_axis(coincidences[0,:,:],ax_idler,colorMap)
    sig = _make_coincidences_axis(coincidences[1,:,:],ax_signal,colorMap)

    ax_signal.set_title("Signal")
    ax_idler.set_title("Idler")

    return (fig, sig, idl)

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


def plot_histogram(coincidences:np.ndarray, min_bin=-200, max_bin=200, color='r', fig:Figure=None) -> Figure:
    if fig is None:
        fig = plt.figure(figsize=(4,8))
        ax = fig.add_axes([0,0,1,1])
    else:
        ax = fig.gca()

    ax.set_xlabel("dt [ns]")
    ax.set_ylabel("Count")
    
    if min_bin > 0: min_bin += 1
    if min_bin <= 0: min_bin -= 1
    if max_bin >= 0: max_bin += 1
    if max_bin < 0: max_bin -= 1
    
    bin_base = (abs(min_bin) + abs(max_bin)) / 12.5
    
    num = bin_base
    i = 1
    while(num < 100):
        num = round(bin_base * (2**i))
        i += 1
        
    #print(f"{num=} = ({bin_base=}) * ({i-1=})")

    bins = xp.linspace(min_bin,max_bin,num)
    # get difference between signal and idler arrival times
    dt = coincidences[1,2,:] - coincidences[0,2,:]

    ax.hist(dt,bins,color=color)

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

    if orientation == 'y':
        data = view[:,loc]
    else:
        data = view[loc,:]

    ax.bar(xp.arange(data.size)+0.5,data,color='gray')

    return (fig,data,view)

def plot_coincidence_xy(correlations:np.ndarray, sign:int=1, \
                        colorMap:str='viridis', fig:Figure=None)\
                                                   -> tuple[Figure,np.ndarray]:
    if fig is None:
        fig = plt.figure(figsize=(4,8))
        ax = fig.add_axes([0,0,1,1])
    else:
        ax = fig.gca()

    data = correlations[0,:,:] + (sign/np.abs(sign)) * correlations[1,:,:]
    view = _make_coincidences_axis(data,ax,colorMap)

    ax.set_xlabel(r'$x_{idl} + x_{sig}$')
    ax.set_ylabel(r'$y_{idl} + y_{sig}$')

    return (fig,view)

def cross_correlation(ref:np.ndarray, target:np.ndarray, flipped=True, plot=True):
    # performs a 2D Pearson CXC
    if flipped:
        target = np.rot90(target, 2)
        
    ref = (ref - np.mean(ref)) / np.std(ref)
    target = (target - np.mean(target)) / np.std(target)

    cxc = signal.correlate(target,ref,mode='same')
    cxc = cxc / cxc.size
    
    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.imshow(asnumpy(ref),origin='lower',aspect="equal",interpolation="none")
        ax2.imshow(asnumpy(target),origin='lower',aspect="equal",interpolation="none")
        ax3.imshow(asnumpy(cxc),origin='lower',aspect="equal",interpolation="none",vmin=0,vmax=1)

        plt.show()

    return cxc

def _make_coincidences_axis(pix:np.ndarray,ax:Axes,\
                           colorMap:str='viridis') -> np.ndarray:
    (view,xrange,yrange) = _make_view(pix)

    if type(colorMap) is str:
        cmap = copy.copy(cm.get_cmap(colorMap))
    else:
        cmap = copy.copy(colorMap)
    if hasattr(cmap,'colors'):
        cmap.set_bad(cmap.colors[0]) #type: ignore
    else:
        cmap.set_bad(cmap(0)) #type: ignore

    ax.imshow(asnumpy(view),origin='lower',aspect='auto',extent=[0,xrange,0,yrange],\
        interpolation='none',cmap=cmap)

    return view

def _make_view(pix:np.ndarray):
    xmin = np.min(pix[0,:])
    xmax = np.max(pix[0,:])
    ymin = np.min(pix[1,:])
    ymax = np.max(pix[1,:])

    xrange = int(xmax - xmin)
    yrange = int(ymax - ymin)

    view = xp.zeros((yrange+1,xrange+1))

    indices = ((pix[1,:] - ymin).astype(int),(pix[0,:]-xmin).astype(int))
    add_at(view,indices,1) # adds 1 to the view value at each hit's (x,y)

    return (view,xrange,yrange)
'''
Contains all of the output viewing functions for use with the TimePix3 camera.
'''

import copy
from tpx3_toolkit.core import Beam, xp, asnumpy, DT
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib import colors as c
import matplotlib.cm as cm
import numpy as np
import scipy.ndimage as snd_np
from scipy import signal as signal_np
try:
    from cupyx.scipy import signal
    import cupyx.scipy.ndimage as snd
except:
    pass

def plot_hits(pix:np.ndarray,
              colorMap:str='viridis',
              fig:Figure=None) -> Figure: #type:ignore

    if fig is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_axes([0,0,1,1])
    else:
        ax = fig.gca()

    make_hits_axes(pix,ax,colorMap)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig

def make_hits_axes(pix:np.ndarray,
                   ax:Axes,
                   colorMap:str='viridis') -> None:

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

def draw_beam_box(ax:Axes,
                  beams:list[Beam],
                  boxColors:list[str]=[]) -> list[LineCollection]:
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

def plot_coincidences(coincidences:np.ndarray,
                      colorMap:str='',
                      fig:Figure=None,
                      ax_signal:Axes=None,
                      ax_idler:Axes=None,) -> Figure: #type: ignore

    if colorMap == '':
        # default red color map to look like a laser idk
        colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
        colorMap = c.LinearSegmentedColormap.from_list(
                "Custom", colors, N=20) #type:ignore

    if fig is None:
        fig = plt.figure(figsize=(12,6))
        ax_signal = fig.add_subplot(122)
        ax_idler = fig.add_subplot(121)
    elif ax_signal is None and ax_idler is None:
        [ax_idler,ax_signal] = fig.axes
            
    idl = _make_coincidences_axis(coincidences[0,:,:],ax_idler,colorMap)
    sig = _make_coincidences_axis(coincidences[1,:,:],ax_signal,colorMap)

    ax_signal.set_title("Signal")
    ax_idler.set_title("Idler")

    return (fig, sig, idl)

def plot_correlations(coincidences:np.ndarray,
                      colorMap:str="gray",
                      fig:Figure=None,
                      ax_x:Axes=None,
                      ax_y:Axes=None) -> Figure: #type:ignore
    if fig is None:
        fig = plt.figure(figsize=(12,6))
        ax_x = fig.add_subplot(121)
        ax_y = fig.add_subplot(122)
    elif ax_x is None and ax_y is None:
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


def plot_histogram(coincidences:np.ndarray, 
                   width=200, # ns
                   num=4, # num of DTs per bin
                   color='r', # bar color
                   fig:Figure=None) -> tuple[Figure,np.ndarray]:
    if fig is None:
        fig = plt.figure(figsize=(4,8))
        ax = fig.add_axes([0,0,1,1])
    else:
        ax = fig.gca()

    ax.set_xlabel("dt [ns]")
    ax.set_ylabel("Count")
    
    spacing = DT * num
    
    width_bin = int(round((width - spacing/2) / spacing))
    min_val = -width_bin * spacing - spacing/2
    max_val = width_bin * spacing + spacing/2
    num_bins = (width_bin + 1) * 2
    
    bins = np.linspace(min_val,max_val,num_bins)
    dt = coincidences[1,2,:] - coincidences[0,2,:]

    vals,_,_ = ax.hist(asnumpy(dt),bins,color=color)

    return fig, vals

def plot_coincidence_trace(pix:np.ndarray, 
                           loc:int, 
                           orientation:str,
                           min_loc:int=0, 
                           max_loc:int=256, 
                           ax:Axes=None) -> tuple[Figure, np.ndarray, np.ndarray]:
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

    ax.bar(np.arange(data.size)+0.5,asnumpy(data),color='gray')

    return (fig,data,view)

def plot_coincidence_xy(correlations:np.ndarray, 
                        sign:int=1,
                        colorMap:str='viridis', 
                        fig:Figure=None) -> tuple[Figure, np.ndarray]:
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

def full_filter_plot(time_filtered_data:np.ndarray,
                     bg_data:np.ndarray) -> tuple[Figure, np.ndarray]:
    # This creates a full filter plot to show all the steps of the filtering 
    # processs in a convenient way

    # need to import this here to avoid a circular import
    from tpx3_toolkit.filter import space_filter_g2
    
    fig = plt.figure(figsize=(6*2,6*5+1))
    axs = fig.subplot_mosaic('''
                            DDEE
                            AABB
                            CCFF
                            GGJJ
                            HHII
                            ''')

    plot_coincidences(time_filtered_data, colorMap='viridis', fig=fig, 
                             ax_signal=axs['D'], ax_idler=axs['E'])
    axs['D'].set_xlabel("$k_x$", fontsize=16)
    axs['D'].set_ylabel("$k_y$", fontsize=16)
    axs['D'].set_title("Direct Signal Momentum", fontsize=16)
    axs['E'].set_xlabel("$k_x$", fontsize=16)
    axs['E'].set_ylabel("$k_y$", fontsize=16)
    axs['E'].set_title("Direct Idler Momentum", fontsize=16)

    plot_correlations(time_filtered_data, colorMap='viridis', fig=fig,
                             ax_x=axs['A'], ax_y=axs['B'])
    axs['A'].set_xlabel("$k_s$", fontsize=16)
    axs['A'].set_ylabel("$k_i$", fontsize=16)
    axs['A'].set_title("Momentum X-Component Correlation", fontsize=16)
    axs['B'].set_xlabel("$k_s$", fontsize=16)
    axs['B'].set_ylabel("$k_i$", fontsize=16)
    axs['B'].set_title("Momentum Y-Component Correlation", fontsize=16)

    fig.sca(axs['C'])
    plot_coincidence_xy(time_filtered_data,fig=fig)
    axs['C'].set_facecolor(colormaps['viridis'](0))
    axs['C'].set_xlabel('$(k_s + k_i)_x$', fontsize=16)
    axs['C'].set_ylabel('$(k_s + k_i)_y$', fontsize=16)
    axs['C'].set_title("Momentum Sum Correlation ($k_s + k_i$)", fontsize=16)
    
    fig.sca(axs['F'])
    plot_coincidence_xy(bg_data,fig=fig)
    axs['F'].set_facecolor(colormaps['viridis'](0))
    axs['F'].set_xlabel('$(k_s + k_i)_x$', fontsize=16)
    axs['F'].set_ylabel('$(k_s + k_i)_y$', fontsize=16)
    axs['F'].set_title("Normalization", fontsize=16)
    
    left = min(axs['C'].get_xlim()[0], axs['F'].get_xlim()[0])
    right = max(axs['C'].get_xlim()[1], axs['F'].get_xlim()[1])
    bottom = min(axs['C'].get_ylim()[0], axs['F'].get_ylim()[0])
    top = max(axs['C'].get_ylim()[1], axs['F'].get_ylim()[1])
    
    space_filtered_data, mask, g_2 = \
        space_filter_g2(time_filtered_data, bg_data)

    axs['G'].imshow(asnumpy(g_2), origin='lower', aspect='equal', 
                    interpolation='none', extent=[left,right,bottom,top])
    axs['G'].set_facecolor(colormaps['viridis'](0))
    axs['G'].set_xlabel('$(k_s + k_i)_x$', fontsize=16)
    axs['G'].set_ylabel('$(k_s + k_i)_y$', fontsize=16)
    axs['G'].set_title("$g(2)$", fontsize=16)

    axs['J'].imshow(asnumpy(np.where(mask,g_2,0)), origin='lower', aspect='equal', 
                    interpolation='none', extent=[left,right,bottom,top])
    axs['J'].set_facecolor(colormaps['viridis'](0))
    axs['J'].set_xlabel('$(k_s + k_i)_x$', fontsize=16)
    axs['J'].set_ylabel('$(k_s + k_i)_y$', fontsize=16)
    axs['J'].set_title("$g(2) > 2$", fontsize=16)

    plot_coincidences(space_filtered_data, colorMap='viridis', fig=fig, 
                             ax_signal=axs['H'], ax_idler=axs['I'])
    axs['H'].set_xlabel("$k_x$", fontsize=16)
    axs['H'].set_ylabel("$k_y$", fontsize=16)
    axs['H'].set_title("Filtered Signal Momentum", fontsize=16)
    axs['I'].set_xlabel("$k_x$", fontsize=16)
    axs['I'].set_ylabel("$k_y$", fontsize=16)
    axs['I'].set_title("Filtered Idler Momentum", fontsize=16)
    
    fig.tight_layout()
    
    return fig, space_filtered_data

def cross_correlation(ref:np.ndarray, 
                      target:np.ndarray, 
                      flipped=True, 
                      plot=True) -> np.ndarray:
    # performs a 2D zero-normalized CXC
    assert type(ref) == type(target), \
    f'''ref and target must have the same type!\n
    {type(ref)=}, {type(target)=}'''
    
    if flipped:
        target = np.rot90(target, 2)
        
    ref = (ref - np.mean(ref)) / np.std(ref)
    target = (target - np.mean(target)) / np.std(target)

    try:
        cxc = signal.correlate2d(target,ref,mode='same')
        cxc = cxc / cxc.size
    except:
        cxc = signal_np.correlate2d(target,ref,mode='same')
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

def bin(view:np.ndarray, xbinsize:int, ybinsize:int=None) -> np.ndarray:
    '''
    Spatially bins the *view* of data.
    '''
    if ybinsize is None: # legacy call
        ybinsize = xbinsize
    
    xcomp = view
    for i in range(xbinsize - 1):
        xcomp += np.roll(view,i,0)
    xcomp = xcomp[::xbinsize,:]

    ycomp = xcomp
    for i in range(ybinsize - 1):
        ycomp += np.roll(xcomp,i,1)
    ycomp = ycomp[:,::ybinsize]

    return ycomp

def magnify_rot(ref:np.ndarray, M:float, rot:float) -> np.ndarray:
    try:
        ref_z = snd.zoom(ref, M)
        ref_z = snd.rotate(ref_z,rot)
    except:
        ref_z = snd_np.zoom(ref, M)
        ref_z = snd_np.rotate(ref_z,rot)
        
    
    ref_z[ref_z < 0] = 0
    
    return ref_z

def _make_coincidences_axis(pix:np.ndarray,
                            ax:Axes,
                            colorMap:str='viridis',
                            flipped=False) -> np.ndarray:
    (view,xrange,yrange) = _make_view(pix)

    if type(colorMap) is str:
        cmap = copy.copy(cm.get_cmap(colorMap))
    else:
        cmap = copy.copy(colorMap)
    if hasattr(cmap,'colors'):
        cmap.set_bad(cmap.colors[0]) #type: ignore
    else:
        cmap.set_bad(cmap(0)) #type: ignore
        
    if flipped:
        view = np.rot90(view,2)

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
    xp.add.at(view,indices,1) # adds 1 to the view value at each hit's (x,y)

    return (view,xrange,yrange)
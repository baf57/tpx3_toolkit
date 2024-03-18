'''
Contains filtering functions for filterng timepix data in potentially helpful
ways.
'''

from tpx3_toolkit.core import xp
from tpx3_toolkit.viewer import add_at
import numpy as np

def time_filter(coincidences:np.ndarray, tmin:float, tmax:float) -> np.ndarray:
    '''
    Filters out coincidences where the time of arrival difference between the
    signal and idler photons is outside of the range defined by [tmin, tmax].
    '''
    dt = coincidences[1,2,:] - coincidences[0,2,:]

    f = (dt >= tmin) & (dt <= tmax)

    return coincidences[:,:,f]

def space_filter(coincidences:np.ndarray, threshold:float) -> np.ndarray:
    '''
    Filters out coincidences where the signal-idler space anti-correlations are
    less than a threshold percentage of the most common signal-idler
    anti-correlations. As long as threshold is low enough, this has the effect 
    of removing accidental coincidences which have no spatial anti-correlation.

    This method runs until convergence since both x and y are done at the same
    time for one iteration, which could lead to the next iteration removing more
    entries.
    '''
    prev = xp.zeros((0,0,0))

    while coincidences.shape[2] != prev.shape[2]:
        xi_min = np.min(coincidences[0,0,:])
        xi_max = np.max(coincidences[0,0,:])
        xs_min = np.min(coincidences[1,0,:])
        xs_max = np.max(coincidences[1,0,:])
        yi_min = np.min(coincidences[0,1,:])
        yi_max = np.max(coincidences[0,1,:])
        ys_min = np.min(coincidences[1,1,:])
        ys_max = np.max(coincidences[1,1,:])

        xi_range = int(xi_max - xi_min)
        xs_range = int(xs_max - xs_min)
        yi_range = int(yi_max - yi_min)
        ys_range = int(ys_max - ys_min)

        x_info = xp.zeros((xs_range+1,xi_range+1))
        y_info = xp.zeros((ys_range+1,yi_range+1))
        x_indices = ((coincidences[1,0,:] - xs_min).astype('int'), \
                (coincidences[0,0,:] - xi_min).astype('int'))
        y_indices = ((coincidences[1,1,:] - ys_min).astype('int'), \
                (coincidences[0,1,:] - yi_min).astype('int'))
        add_at(x_info,x_indices,1)
        add_at(y_info,y_indices,1)

        x_max = np.max(x_info, axis=None)
        y_max = np.max(x_info, axis=None)

        x_mask = x_info > x_max * threshold
        y_mask = y_info > y_max * threshold

        x_filter = x_mask[x_indices]
        y_filter = y_mask[y_indices]

        f = x_filter & y_filter
        prev = coincidences
        coincidences = coincidences[:,:,f]

    return coincidences

def bin(coincidences:np.ndarray, xbins:int, ybins:int) -> np.ndarray:
    '''
    Spatially bins the data, functionally lowering the resolution.
    '''
    coincidences[:,0,:] = np.ceil(coincidences[:,0,:] / xbins)
    coincidences[:,1,:] = np.ceil(coincidences[:,1,:] / ybins)

    return coincidences

def space_filter_alt(coincidences:np.ndarray,threshold:float):
    '''
    Establishes a spatial filter which filters the spatial correlations by a
    percentage of the maximum spatial mode (x and y at the same time in the 
    psuedo-4D (xi+xs, yi+ys) space).
    '''
    data = coincidences[0,:,:] + coincidences[1,:,:]
    
    xmin = np.min(data[0,:])
    xmax = np.max(data[0,:])
    ymin = np.min(data[1,:])
    ymax = np.max(data[1,:])

    xrange = int(xmax - xmin)
    yrange = int(ymax - ymin)

    view = xp.zeros((yrange+1,xrange+1))

    indices = ((data[1,:] - ymin).astype('int'),(data[0,:]-xmin).astype('int'))
    add_at(view,indices,1) # adds 1 to the view value at each hit's (x,y)

    mask = view > (np.max(view) * threshold)

    f = mask[indices]

    return (coincidences[:,:,f], mask)
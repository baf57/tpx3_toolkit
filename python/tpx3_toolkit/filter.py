'''
Contains filtering functions for filterng timepix data in potentially helpful
ways.
'''

from typing import Callable, Union
from tpx3_toolkit.core import xp
from tpx3_toolkit.viewer import cross_correlation, _make_view
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
        xp.add.at(x_info,x_indices,1)
        xp.add.at(y_info,y_indices,1)

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

def space_filter_alt(coincidences:np.ndarray,
                     threshold:float) -> tuple[np.ndarray, np.ndarray]:
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
    xp.add.at(view,indices,1) # adds 1 to the view value at each hit's (x,y)

    mask = view > (np.max(view) * threshold)

    f = mask[indices]

    return (coincidences[:,:,f], mask)

def space_filter_internal(coincidences:np.ndarray,
                          weights:tuple[int,int] = (1,6)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = coincidences[0,:,:] + coincidences[1,:,:]
    
    xmin = np.min(data[0,:])
    xmax = np.max(data[0,:])
    ymin = np.min(data[1,:])
    ymax = np.max(data[1,:])

    xrange = int(xmax - xmin)
    yrange = int(ymax - ymin)

    view = np.zeros((yrange+1,xrange+1))

    indices = ((data[1,:] - ymin).astype('int'),(data[0,:]-xmin).astype('int'))
    np.add.at(view,indices,1) # adds 1 to the view value at each hit's (x,y)
    
    bg = view[150:160,150:160]
    
    mask = (view - (weights[0]*np.sqrt(bg.mean())+
                     (weights[1]*np.sqrt(bg.var(ddof=1))))) > 0

    f = mask[indices]

    return (coincidences[:,:,f], mask, bg)

def space_filter_g2(coincidences:np.ndarray,
                    background:np.ndarray,
                    lower_limit:float = 2.0,
                    upper_limit: Union[float, None] = None) \
                        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perfoms a filter based off of the statistical independence test as defined
    by the "g2" metric. The upper_limit is optional in case I decide to use it 
    to avoid artifacts.
    '''
    view_g2, indices_sum = g2(coincidences,background)
    
    mask = (view_g2 > lower_limit) & np.isfinite(view_g2)
    if upper_limit is not None:
        mask = mask & (view_g2 < upper_limit)
        
    f = mask[indices_sum]
    
    return (coincidences[:,:,f], mask, view_g2)

def best_space_filter(coincidences: np.ndarray, 
                      ref: np.ndarray,
                      precision: float = 0.01,
                      filter: Callable[[np.ndarray, float],
                                       tuple[np.ndarray,np.ndarray]] = space_filter_alt,
                      which: int = 0) -> tuple[float, np.ndarray]:
    '''
    Finds the best space filter for a given data set based on the similarity
    with a reference view. The percision sets an upper bound for the percision
    of the threshold value.
    '''
    # I tried a lot of things here to make it more efficient (binary, gradient 
    # decent, annealing, etc.)... this is a really resilient probelm. I am just
    # going to do the easiest thing and accept that it'll be slow.
    if which == 0:
        flipped = True
    else:
        flipped = False
    
    N = int(np.ceil(1/precision))
    
    threshs = np.linspace(0.0, 1.0, num = N+1)
    
    vals = np.zeros(N+1)
    outs = []
    for i,thresh in enumerate(threshs):
        try:
            (out,_) = filter(coincidences, thresh)
            (view,_,_) = _make_view(out[which,:,:])
            cxc = cross_correlation(ref, view, flipped, plot=False)
            outs.append(out)
            vals[i] = float(cxc.max())
        except:
            outs.append(np.nan)
            vals[i] = np.nan
            
    thresh_out = float(threshs[np.nanargmax(vals)])
    thresh_out = round(thresh_out, -round(np.log10(precision))) # rounds to precision
        
    return (thresh_out, outs[np.nanargmax(vals)])

def g2(coincidences: np.ndarray,
       background: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    data_i = coincidences[0,:,:] # idler events (singles)
    data_s = coincidences[1,:,:] # signal events (singles)
    bg_i = coincidences[0,:,:] # idler background events (singles)
    bg_s = coincidences[1,:,:] # signal background events (singles)
    data_sum = data_i + data_s # idler + signal (paired AND summed) events
    bg_sum = bg_i + bg_s # idler + signal background events
    
    #xmin_i = np.min(data_i[0,:])
    #xmax_i = np.max(data_i[0,:])
    #ymin_i = np.min(data_i[1,:])
    #ymax_i = np.max(data_i[1,:])
    
    #xmin_s = np.min(data_s[0,:])
    #xmax_s = np.max(data_s[0,:])
    #ymin_s = np.min(data_s[1,:])
    #ymax_s = np.max(data_s[1,:])
    
    xmin_sum = np.min(data_sum[0,:])
    xmax_sum = np.max(data_sum[0,:])
    ymin_sum = np.min(data_sum[1,:])
    ymax_sum = np.max(data_sum[1,:])
    
    xmin_bg_sum = np.min(bg_sum[0,:])
    xmax_bg_sum = np.max(bg_sum[0,:])
    ymin_bg_sum = np.min(bg_sum[1,:])
    ymax_bg_sum = np.max(bg_sum[1,:])
    
    xrange = int(xmax_sum - xmin_sum)
    yrange = int(ymax_sum - ymin_sum)
    
    #view_i = np.zeros((xrange+1,yrange+1)) # <I(k_xi, k_yi)>
    #view_s = view_i.copy()                 # <I(k_xs, k_ys)>
    view_sum = xp.zeros((xrange+1,yrange+1))# <I(k_xi+k_xs, k_yi+k_ys)>
    view_bg = view_sum.copy()            # <I(k_xi+k_xs, k_yi+k_ys)>_uc

    #indices_i = (((data_i[0,:] - xmin_i) + ((xrange+1) / 4)).astype('int'), # k_xi
    #             ((data_i[1,:] - ymin_i) + ((yrange+1) / 4)).astype('int')) # k_yi
    #indices_s = (((data_s[0,:] - xmin_s) + ((xrange+1) / 4)).astype('int'), # k_xs
    #             ((data_s[1,:] - ymin_s) + ((yrange+1) / 4)).astype('int')) # k_ys
    indices_sum = ((data_sum[1,:] - ymin_sum).astype('int'), # k_yi + k_ys
                   (data_sum[0,:] - xmin_sum).astype('int')) # k_xi + k_ys
    indices_bg_sum = ((bg_sum[1,:] - ymin_sum).astype('int'), # (k_yi + k_ys)_uc
                      (bg_sum[0,:] - bg_sum).astype('int')) # (k_xi + k_ys)_uc
    
    #np.add.at(view_i,indices_i,1) # adds 1 to the view value at each hit's (x,y)
    #np.add.at(view_s,indices_s,1)
    xp.add.at(view_sum,indices_sum,1)
    xp.add.at(view_bg,indices_bg_sum,1)
    
    #view_i_x = np.sum(view_i,axis=0) / (xrange+1) # <I(k_xi)>
    #view_i_y = np.sum(view_i,axis=1) / (yrange+1) # <I(k_yi)>
    #view_s_x = np.sum(view_s,axis=0) / (xrange+1) # <I(k_xs)>
    #view_s_y = np.sum(view_s,axis=1) / (yrange+1) # <I(k_ys)>
    
    #norm_x = view_i_x + view_s_x # <I(k_xi)> + <I(k_xs)>
    #norm_y = view_i_y + view_s_y # <I(k_yi)> + <I(k_ys)>
    ## <I(k_xi, k_xs)> + <I(k_yi, k_ys)>
    #norm = np.outer(norm_x, norm_y) / ((norm_x.max() + norm_y.max()) / 2)
    
    with np.errstate(divide='ignore'):
        # <I(k_xi+k_xs, k_yi+k_ys)> / (<I(k_xi, k_xs)> + <I(k_yi, k_ys)>)
        view = view_sum / view_bg
    view[view_bg==0] = np.nan
    
    return view, indices_sum
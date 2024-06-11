'''
Contains filtering functions for filterng timepix data in potentially helpful
ways.
'''

from typing import Callable, Union
from tpx3_toolkit.core import xp, asnumpy
from tpx3_toolkit.viewer import cross_correlation, _make_view
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter,generic_filter

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
                    upper_limit: Union[float, None] = None,
                    norm_scale: float = 1.0,
                    norm_cutoff: float = 1.0,
                    norm_fit: bool = True,
                    norm_smooth: bool = False,
                    neighbors: bool = False,
                    neighbor_distance: int = 7) \
                        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perfoms a filter based off of the statistical independence test as defined
    by the "g2" metric. The upper_limit is optional in case I decide to use it 
    to avoid artifacts.
    '''
    view_g2, indices_sum = g2(coincidences,background,norm_cutoff,norm_scale,
                              norm_fit, norm_smooth)
    
    mask = (view_g2 > lower_limit) & np.isfinite(view_g2)
    if upper_limit is not None:
        mask = mask & (view_g2 < upper_limit)
        
    if neighbors:
        mask = _g2_neighbors(mask, neighbor_distance) # another type of smoothing essentially
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
       background: np.ndarray,
       cutoff: float = 1.0,
       norm_scale: float = 1.0,
       fit: bool = True,
       smooth: bool = False) -> tuple[np.ndarray,np.ndarray]:
    data_i = coincidences[0,:,:] # idler events (singles)
    data_s = coincidences[1,:,:] # signal events (singles)
    bg_i = background[0,:,:] # idler background events (singles)
    bg_s = background[1,:,:] # signal background events (singles)
    data_sum = data_i + data_s # idler + signal (paired AND summed) events
    bg_sum = bg_i + bg_s # idler + signal background events
    
    xmin_sum = np.min(data_sum[0,:])
    xmax_sum = np.max(data_sum[0,:])
    ymin_sum = np.min(data_sum[1,:])
    ymax_sum = np.max(data_sum[1,:])
    
    xmin_bg_sum = np.min(bg_sum[0,:])
    xmax_bg_sum = np.max(bg_sum[0,:])
    ymin_bg_sum = np.min(bg_sum[1,:])
    ymax_bg_sum = np.max(bg_sum[1,:])
    
    xrange_sum = int(xmax_sum - xmin_sum)
    yrange_sum = int(ymax_sum - ymin_sum)
    xrange_bg_sum = int(xmax_bg_sum - xmin_bg_sum)
    yrange_bg_sum = int(ymax_bg_sum - ymin_bg_sum)
    xrange_diff = xrange_bg_sum - xrange_sum
    yrange_diff = yrange_bg_sum - yrange_sum
    
    # <I(k_xi+k_xs, k_yi+k_ys)I(k_xi+k_xs, k_yi+k_ys)>
    view_sum = xp.zeros((yrange_sum+1,xrange_sum+1))
    # <I(k_xi+k_xs, k_yi+k_ys)><I(k_xi+k_xs, k_yi+k_ys)>
    view_bg = xp.zeros((yrange_bg_sum+1,xrange_bg_sum+1))

    indices_sum = ((data_sum[1,:] - ymin_sum).astype('int'), # k_yi + k_ys
                   (data_sum[0,:] - xmin_sum).astype('int')) # k_xi + k_ys
    indices_bg_sum = ((bg_sum[1,:] - ymin_bg_sum).astype('int'), # (k_yi + k_ys)_uc
                      (bg_sum[0,:] - xmin_bg_sum).astype('int')) # (k_xi + k_ys)_uc
    
    xp.add.at(view_sum,indices_sum,1)
    xp.add.at(view_bg,indices_bg_sum,1)
    
    # DEBUG for size issue if it ever comes up   
    #print(f'Before: {view_sum.shape=}, {view_bg.shape=}')
    
    if xrange_diff > 0: # bg_x > sum_x -> make bg_x smaller
        xlow = int(xrange_diff / 2)
        xhigh = -xlow
        if xrange_diff % 2 == 1: # xrange_diff is odd
            xlow += 1
            
        if xhigh == 0: # why not be consistent with -0 indexing NumPy ???
            view_bg = view_bg[:,xlow:]
        else:
            view_bg = view_bg[:,xlow:xhigh]
    elif xrange_diff < 0: # bg_x < sum_x -> make bg_x bigger
        padleft = int(-xrange_diff/2)
        padright = padleft
        if xrange_diff % 2 == 1:
            padright += 1
            
        view_bg = np.pad(view_bg,
                  ((0,0),
                   (padleft,padright)),
                  mode='edge') # pads with edge to not mess up the fit much
        
    if yrange_diff > 0: # bg_y > sum_y -> make bg_y smaller
        ylow = int(yrange_diff / 2)
        yhigh = -ylow
        if yrange_diff % 2 == 1: # yrange_diff is odd
            ylow += 1
           
        if yhigh == 0:
            view_bg = view_bg[ylow:,:]
        else: 
            view_bg = view_bg[ylow:yhigh,:]
    elif yrange_diff < 0: # bg_y < sum_y -> make bg_y bigger
        padbot = int(-yrange_diff/2)
        padtop = padbot
        if yrange_diff % 2 == 1:
            padbot += 1
            
        view_bg = np.pad(view_bg,
                  ((padtop,padbot), # top <-> bot
                   (0,0)),
                  mode='edge') # pads with edge to not mess up the fit much
    
    # DEBUG for size issue if it ever comes up   
    #print(f'After: {view_sum.shape=}, {view_bg.shape=}')
    view_bg = view_bg * norm_scale
    
    if fit:
        view_bg = _fit_normalization(view_bg) # throwing a fit
    view_bg[view_bg<cutoff] = cutoff # this prevents explosive values
    if smooth:
        view_bg = _smooth_correlations(view_bg,3) # smooths the correlations
            
    with np.errstate(divide='ignore'):
        # <I(k_xi+k_xs, k_yi+k_ys)I(k_xi+k_xs, k_yi+k_ys)> / 
        # <I(k_xi+k_xs, k_yi+k_ys)><I(k_xi+k_xs, k_yi+k_ys)>
        view = view_sum / view_bg
    
    return view, indices_sum

def _fit_normalization(background):
    # CuPy has no 'curve_fit' function, and I don't feel like implementing it
    # using polyfit, so the easy solution is to just convert to numpy then back
    # to CuPy. These arrays are small anyway
    def gen_2dgauss(A, x0, y0):
        def two_gauss(xy, sigma_x, sigma_y, theta, P):
                x, y = xy
                a = (np.cos(theta)**2)/(2*sigma_x**2) + \
                    (np.sin(theta)**2)/(2*sigma_y**2)
                b = -(np.sin(2*theta))/(4*sigma_x**2) + \
                    (np.sin(2*theta))/(4*sigma_y**2)
                c = (np.sin(theta)**2)/(2*sigma_x**2) + \
                    (np.cos(theta)**2)/(2*sigma_y**2)
                    
                B = a * ((x-x0)**2)
                C = 2*b * (x-x0)*(y-y0)
                D = c * ((y-y0)**2)
                
                g = A*np.exp(-(B + C + D)**P)
                
                return g.ravel()
        return two_gauss

    background = asnumpy(background)
    x_bg = np.arange(background.shape[1])
    y_bg = np.arange(background.shape[0])
    xy_bg = np.meshgrid(x_bg, y_bg)
    xx_bg, yy_bg = xy_bg
    x0 = np.average(xx_bg, weights=background)
    y0 = np.average(yy_bg, weights=background)
    A = background[np.sqrt((x0-xx_bg)**2 + (y0-yy_bg)**2)<10].mean()
    
    bg_gauss = gen_2dgauss(A, x0, y0)
    popt, _ = curve_fit(bg_gauss, xy_bg, background.ravel())
    bg_fit = xp.array(bg_gauss(xy_bg, *popt).reshape(xy_bg[0].shape))
    
    return bg_fit

def _smooth_correlations(correlations,strength):
    correlations = gaussian_filter(asnumpy(correlations),strength)
    return xp.array(correlations)

def _g2_neighbors(correlations, neighbor_distance):
    def check_neighbors(array):
        if array[2] == 1:
            return array.mean() > 0.8
        return False
    
    limit = int((np.ceil((neighbor_distance + 1) / 2) * 2 - 1))
    
    ul = np.zeros((int(np.ceil(limit/2)),int(np.ceil(limit/2))))
    ul[0,:] = np.linspace(limit/2, int(limit/2), num = ul.shape[0])
    i = int(ul[0, -1])
    for i,j in enumerate(np.arange(i)[::-1]):
        ul[i+1,:] = np.linspace(ul[0,i+1], j, num = ul.shape[0])
    
    footprint = np.concatenate([ul,ul[:,-2::-1]], axis = 1)
    footprint = np.concatenate([footprint, footprint[-2::-1,:]], axis = 0)
    
    footprint = np.where(footprint <= neighbor_distance/2, 1, 0)
    correlations = generic_filter(asnumpy(correlations), check_neighbors, 
                                  footprint=footprint)
    
    return xp.array(correlations)
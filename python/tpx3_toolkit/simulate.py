'''
Contains simulation functions for making fake TPX3 or coincidence data. The 
majority of this submodule likely cannot be done in parallel as the typical 
sizes of added datat well exceed typical VRAM amounts.
'''

from tpx3_toolkit.core import Beam, DT, xp, asnumpy
from scipy.sparse import random as sp_random
from scipy.special import jacobi
from scipy.stats import (multivariate_normal as mvn)
from typing import Union, Literal
import numpy as np

def add_coherent(data: np.ndarray, 
                 num_hits: int, 
                 beams: list[Beam],
                 circular_beam: bool = True,
                 time: Union[float,None]=None,
                 verbose: bool = False) -> tuple[np.ndarray, int]:
    '''
    Adds simulated hits to an existsing pix array, within the bounds of said 
    existing array.
    
    Parameters 
    ----------
    data: ndarray or None
        a pix array as described in t3.core.parse_raw_file(). If None is given,
        then a new array will be created
    num_hits: int
        the target number of hits to add. This number of hits may not be exactly
        added, but will be used to calculated the expectation value of hits in 
        each temporal mode
    beams: list[Beam]
        a list of beams which describes where to add the hits. The hits will be
        (roughly) evenly distributed between the beams if multiple are given
    circular_beam: bool, optional, default=True
        if True, then hits are added in an oval which best fits within each
        given beam. Otherwise the hits are added in the entire beam area
    time: float or None, optional, default=None
        the amount of time in seconds over which the events should be added. If
        data is None, then this time must be set. Otherwise this will be
        determined by the min and max ToA of data.
    verbose: bool, optional, default=False
        if True, progress info will be printed to the console. This is generally
        a long process, so such info can be helpful
        
    Returns
    -------
    extended_data: np.ndarray
        the original data, now with the added simulated hits. Still pix-like
    hits_added: int
        the number of added hits
    
    Notes
    -----
    Hits are added probabilistically, and so the `num` parameter only
    defines a target for the number of added hits. It is used as a target to
    calculate the expectation value per timebin for sampling the poissonian
    distribution. The positions of the photons are normaly distributed, as 
    with the ToT.

    These are only noise hits, and have no intended correlation with eachother 
    whatsoever. This is equivalent to an external source of uncorrelated light
    following poissonian statistics.
    '''
    if num_hits == 0:
        return data, 0
    
    if data is None:
        assert time is not None, f'If creating a new data array, then the ' + \
                                 'time parameter must not be None'
        toa_bounds = (0, time*10**9)
        tot_bounds = (0, 0)
    else:
        maxs = data.max(axis=1)
        mins = data.min(axis=1)
        
        toa_bounds = (float(mins[2]), float(maxs[2]))
        tot_bounds = (float(maxs[3]), float(mins[3]))
    
    n_bins = int((max(toa_bounds) - min(toa_bounds)) / DT)
    new_n_exp = num_hits / n_bins

    try: # would prefer to change xp, but Python does not allow this, thus flag
        expected_size = n_bins * 64 + num_hits * 64 # bytes (8 bits, 2x concat size, 4 fields)
        free_bytes = xp.cuda.Device(0).mem_info[1]
        if expected_size >= free_bytes:
            print(f'~{expected_size/(2**30):.2f}GiB of VRAM required, but ' + \
                  f'only {free_bytes/(2**30):.2f}GiB availalbe -> Using CPU.')
            CUDA = False
        else:
            CUDA = True
    except:
        CUDA = False
    
    new_hits = _gen_hits(new_n_exp, 
                        n_bins,
                        beams, 
                        toa_bounds, 
                        tot_bounds,
                        circular_beam,
                        verbose,
                        CUDA)

    if verbose: print(f'\thits generated, concatenating...')

    # CUDA output if possible
    if data is not None:
        data_out = np.concatenate([data,xp.asarray(new_hits)], axis=1)
    else:
        data_out = xp.array(new_hits)

    hits_added = new_hits.shape[1]
    
    if verbose: print(f'Done concatenaing! {hits_added} hits added')
    
    try:
        xp.get_default_memory_pool().free_all_blocks()
    except:
        # cp not being used
        pass

    return data_out, hits_added

def add_SPDC(data: Union[np.ndarray,None], 
             num_pairs: int, 
             dToA_var: float,
             linear_corr_strength: float,
             beams: list[Beam],
             time: Union[float, None] = None,
             verbose: bool = False) -> tuple[np.ndarray, int]:
    '''
    Adds pairs SPDC events to an existing pix-like array, or generates a new
    array.
    
    Parameters
    ----------
    data: ndarray or None
        a pix array as described in t3.core.parse_raw_file(). If None is given,
        then a new array will be created
    num_pairs: int
        the target number of hit pairs to add. This number of pairs may not be
        exactly added, but will be used to calculated the expectation value of
        pairs in each temporal mode
    dToA_var: float
        the variance in the ToA difference (nanoseconds) between photons in a
        pair. Generally this will be a Gaussian distribution around the central
        ToA time
    linear_corr_strength: float
        the strength of a the linear (anit-)correlation between the singal and 
        idler positions in the far field (techincally momentum difference). This
        value should be between 1 (completely anti-correlated) and 0 (no linear
        correlation). A value of 1 describes a perfectly anti-correlated SPDC 
        source, while a value of 0 describes essentially two synced pulse 
        lasers (with <<1 average photons per pulse).
    beams: list[Beam]
        a list of beams which describes where to add the hits. If one beam is 
        given, then the hits will both be distributed throughout the beam. If 
        two are given, then the first beam will be treated as the idler, while
        the second beam will be treated as the signal
    time: float or None, optional, default=None
        the amount of time in seconds over which the events should be added. If
        data is None, then this time must be set. Otherwise this will be
        determined by the min and max ToA of data.
    verbose: bool, optional, default=False
        if True, progress info will be printed to the console. This is generally
        a long process, so such info can be helpful
        
    Returns
    -------
    data_out: np.ndarray
        if data is given, then this is the original data but now with the added 
        simulated pairs. If no data was given then this is just the simulated 
        pairs. Either way it is pix-like
    pairs_added: int
        the number of added pairs
    
    Notes
    -----
    Pais are added probabilistically, and so the `num` parameter only
    defines a target for the number of added pairs. It is used as a target to
    calculate the expectation value per timebin for sampling the poissonian
    distribution. The positions of the photons are normaly distributed around
    the center, but correlated. The ToT tends to not really matter at this 
    point, so it is just generated as 0 for all hits for now.
    '''
    if num_pairs == 0:
        return data, 0
    
    if data is None:
        assert time is not None, f'If creating a new data array, then the ' + \
                                 'time parameter must not be None'
        toa_bounds = (0, time*10**9)
        tot_bounds = (0, 0)
    else:
        maxs = data.max(axis=1)
        mins = data.min(axis=1)
        
        toa_bounds = (float(mins[2]), float(maxs[2]))
        tot_bounds = (float(maxs[3]), float(mins[3]))
        
    n_bins = int((toa_bounds[1] - toa_bounds[0]) / DT)
    # increasing number of pairs to acconut for the ones that may fall outside
    # the beam when generating. I find that ~97% of the hits are preserved, so 
    # that is how I chose this scale. It's arbitrary-ish, but it works
    new_n_exp = int(num_pairs/0.97) / n_bins
    
    try: # would prefer to change xp, but Python does not allow this, thus flag
        expected_size = n_bins * 64 # bytes (8 bits, 2x concat size, 4 fields)
        free_bytes = xp.cuda.Device(0).mem_info[1]
        if expected_size >= free_bytes:
            print(f'~{expected_size/(2**30):.2f}GiB of VRAM required, but ' + \
                  f'only {free_bytes/(2**30):.2f}GiB availalbe -> Using CPU.')
            CUDA = False
        else:
            CUDA = True
    except:
        CUDA = False
    
    #if verbose: print(f'calculated n_exp = {new_n_exp:.4f}')
    
    new_hits = _gen_pairs(new_n_exp,
                          n_bins,
                          dToA_var,
                          beams,
                          toa_bounds,
                          tot_bounds,
                          linear_corr_strength,
                          verbose,
                          CUDA)
    
    if verbose: print(f'hits generated, concatenating to old data...')

    # CUDA output if possible
    if data is not None:
        data_out = np.concatenate([data,xp.asarray(new_hits)], axis=1)
    else:
        data_out = xp.array(new_hits)
            
    hits_added = new_hits.shape[1]
    
    if verbose: print(f'\nDone concatenaing! {hits_added} events ({hits_added/2:.0f} pairs) added')
    
    try:
        xp.get_default_memory_pool().free_all_blocks()
    except:
        # cp not being used
        pass

    return data_out, hits_added

def apply_target_mask(data: np.ndarray, 
                      mask: np.ndarray, 
                      beam: Beam):
    '''
    Applies a binary target mask to a region of data defined by a beam.
    
    Parameters
    ----------
    data: ndarray
        The data on which this beam masking will be performed. Format of a pix
        type array (see core.parse_raw_file()).
    mask: ndarray
        The binary target mask to apply. Where it is true it will let through
        (keep) hits, otherwise it will block (remove) them.
    beam: Beam
        The region in which to apply the mask. The mask and the beam must be the
        same size and shape.
        
    Returns
    -------
    data: ndarray
        The same input data with hits removed.
    num_removed: int
        The number of hits removed by the masking.
    '''
    assert beam.area == mask.size, \
        f"Mask must be same size as beam! ({mask.size}!={beam.area})"
        
    loc = data[:2,:].copy()
    loc[0,:] = loc[0,:] - beam.left
    loc[1,:] = loc[1,:] - beam.bottom
    
    f = np.where(beam.in_beam(data), 
                 mask[tuple(loc.astype(int))],
                 True)
    
    return data[:,f], f.size - f.sum()
    
def distort_beam(data:np.ndarray,
                 distance: int,
                 strength: float,
                 uniformity: float = 0.25,
                 fineness: int = 1,
                 maps_in: Union[list[np.ndarray],None] = None,
                 keep_in_beam: bool = True):
    # data: PIX type
    # distance: integer > 0
    # strength: 0 < float < 1
    # fineness: integer >= 1
    # region: Beam or None -> just one Beam for now
    '''
    THIS IS EXPERIMENTAL, AND CURRENTLY RUNS VERY SLOWLY, SEE NOTES.
    Distorts a beam in data to simulate propagating throuhg turbulent media. 
    
    Parameters
    ----------
    data: ndarray
        The data on which the distortion will be performed. Format of a pix type
        array (see core.parse_raw_file()).
    distance: int
        The max distance to displace an event from the origin. Must be greater 
        than 0. To prevent unexpected behaviour, the bounds of the data should
        be less than (distance * 2) + 1.
    strength: float
        The likelihood that any individual event will be displaced. Must be
        between 0 (no displacement will occur) and 1 (displacement will always
        occur).
    uniformity: float, default = 0.25
        The uniformity of the displacement. Must be between 0 (no displacement)
        and 1 (uniform displacement over the defined distance region). This 
        value is contrained by strength (i.e. a stength of 0.25 and uniformity
        of 1 would mean that the event will be displaced 25% of the time, but
        when it is diplaced it is done so uniformly)
    fineness: int, default = 1
        The frequency at which a new displacement transformation is calculated.
        Must be between 1 (only 1 transformation for all input modes), and the 
        size of the x-y spread of the data input array. WARNING: this variable 
        can greatly effect the memory usage of this function.
    maps_in: ndarray or None, default = None 
        If previously calculated maps are desired to be used (e.g. distorting 
        the data filtered in one way, and then desiring to distort the same
        data but filtered in an alternative way), this is where to provide them
        as input. This will overwrite the distance, strength, uniformity, and 
        fineness metrics, though the fineness must correspond properly with the
        number of provided maps (i.e. sqrt(len(maps_in)) == fineness).
    keep_in_beam: bool, default = True
        A flag that determines what to do with events which are displaced out of
        the original bounds defined by the data. If False, all values will be
        left outside the bounds of data (this may cause negative x/y values, so
        be warned). If True, events which displace outside the bounds of the
        data will be reflcted over the edge of the data back into the data
        bounds. If the upper constraint on distance described above is not 
        followed, it is possible for events to still fall outside the bounds of
        the data, even if this value is set to True
        
    Returns
    -------
    new_data: ndarray
        The displaced data. Will be the same size as the input data.
    maps: list[ndarray]
        The maps used to calculate the displacements.
        
    Notes
    -----
    This is inspired by the definition of a quantum process, where every input
    mode will experience some unitary transformation which maps it to some
    subset of the output modes. As such, multiple unitary transofrmations can be
    generated to be applied to every input mode. By default, only one unitary is
    calculated. This is for simplicity, and because it seems to work well.
    Generally though this could have up to n*m unitaries generated, when there 
    are n*m input modes. In this case however, the memory consumption would be 
    huge.
    
    Note that this is only simulated for intensity, and as such all unitary
    transformations mentioned here are real-valued.
    
    This function is very slow in its current state. The function could be 
    vectorized, however doing so makes it EXTREMELY memory intensive (even
    for distorting only 100000 events this can require up to 50GB for a distance
    of 50 pixels, for instance). As such, I am currently leaving it 
    unvectortized. In the future I think if I were to attempt to use this more I
    would make a switch which would esitmate the needed memory, and if that 
    amount exceeded the available memory then I switch to sequential.    
    '''
    xmin = int(data[0,:].min())
    xmax = int(data[0,:].max())
    ymin = int(data[1,:].min())
    ymax = int(data[1,:].max())
    xrange = xmax - xmin
    yrange = ymax - ymin
    xnum = xrange // fineness
    ynum = yrange // fineness
    map_shape = ((2*distance)+1,(2*distance)+1)
    
    if maps_in is None:
        maps = []
    else:
        maps = maps_in
        
    key = xp.zeros((xrange,yrange))
    for i in range(fineness**2):
        if maps_in is None:
            curr = sp_random(map_shape[0], map_shape[1], density=uniformity).toarray()
            curr = (curr / curr.sum()) * (strength)
            curr[distance, distance] = 1 - strength
            curr = np.cumsum(curr)
            maps.append(xp.array(curr))
        
        xlo = xnum * (i % fineness)
        xhi = xnum * ((i % fineness) + 1)
        ylo = ynum * (i // fineness)
        yhi = ynum * ((i // fineness) + 1)
        key[xlo:xhi,ylo:yhi] = i
        
    new_data = data.copy()
    rands = xp.random.rand(data.shape[1])
    print("Calcualting offset...") # can't vectorize without repeating map
    for i in range(data.shape[1]): # rand.size number of times, which is huge :(
        if(i%1000 == 0):
            print(f'\tprogress: {i / data.shape[1] * 100:5.1f}%', end='\r')
        index = np.argmax(\
                        maps[int(key[tuple(data[:2,i].astype(int))])] > rands[i])
        offset = np.unravel_index(int(index),map_shape)
        offset = xp.array([offset[0] - distance,offset[1] - distance])
        new_data[:2,i] = data[:2,i] + offset
    print('\tprogress: 100.0%\nDone!')
    
    if keep_in_beam:
        new_data[0,new_data[0,:]<data[0,:].min()] = data[0,:].min() + \
            (data[0,:].min() - new_data[0,new_data[0,:]<data[0,:].min()])
        new_data[0,new_data[0,:]>data[0,:].max()] = data[0,:].max() - \
            (new_data[0,new_data[0,:]>data[0,:].max()] - data[0,:].max())

        new_data[1,new_data[1,:]<data[1,:].min()] = data[1,:].min() + \
            (data[1,:].min() - new_data[1,new_data[1,:]<data[1,:].min()])
        new_data[1,new_data[1,:]>data[1,:].max()] = data[1,:].max() - \
            (new_data[1,new_data[1,:]>data[1,:].max()] - data[1,:].max())
    
    return new_data, maps

def distort_beam_zernike(data:np.ndarray,
                         j_max: int,
                         tightness: float,
                         spread: float,
                         circle: bool = False,
                         Z_in: np.ndarray = None,
                         keep_in_beam: bool = True):
    '''
    THIS IS EXPERIMENTAL, AND CURRENTLY RUNS VERY SLOWLY, SEE NOTES.
    Distorts a beam in data to simulate propagating through turbulent media, 
    based on Zernike polynomials.
    
    Parameters
    ----------
    data: ndarray
        The data on which the distortion will be performed. Format of a pix type
        array (see core.parse_raw_file()).
    j_max: int
        The numer of Zernike terms to condsider in the calculation of the 
        polynomial. Must be between 1 and 15 (inclusive).
    tightness: float
        The tightness of the displacement. Must be between 0 (no tightness) and
        less than 1 (absolutely tight).
    spread: int
        The max distance to displace an event from the origin. Must be greater 
        than 0.
    circle: bool, default = False
        If true, constrains the Zernike polynomial to a circle bounded by the
        min and max of data_x and data_y. 
    Z_in: ndarray or None, default = None 
        If a previously calculated Zernike polynomial is desired to be used
        (e.g. distorting the data filtered in one way, and then desiring to
        distort the same data but filtered in an alternative way), this is where
        to provide it as input. This will overwrite the other parameters, though
        the spread must match the size of Z_in.
    keep_in_beam: bool, default = True
        A flag that determines what to do with events which are displaced out of
        the original bounds defined by the data. If False, all values will be
        left outside the bounds of data (this may cause negative x/y values, so
        be warned). If True, events which displace outside the bounds of the
        data will be reflcted over the edge of the data back into the data
        bounds. If the upper constraint on distance described above is not 
        followed, it is possible for events to still fall outside the bounds of
        the data, even if this value is set to True
        
    Returns
    -------
    new_data: ndarray
        The displaced data. Will be the same size as the input data.
    Z: list[ndarray]
        The Zernike polynomial used to calculate the displacements.
        
    Notes
    -----
    This is inspired by the abbaration of a light beam as can be modeled by a
    Zernike polynomial. 
    
    Note that this is only simulated for intensity, and as such interference 
    effects are not considered.
    
    This function is very slow in its current state. The function could be 
    vectorized, however doing so makes it very memory intensive (even for
    distorting only 500000 events this can require up to 8GB for a spread of 50
    pixels, for instance). As such, I am currently leaving it unvectortized. In
    the future I think if I were to attempt to use this more I would make a
    switch which would esitmate the needed memory, and if that amount exceeded
    the available memory then I switch to sequential.    
    '''
    xmin = int(data[0,:].min())
    xmax = int(data[0,:].max())
    ymin = int(data[1,:].min())
    ymax = int(data[1,:].max())
    
    if Z_in is None:
        #x = np.arange(xmin,xmax+1)
        #y = np.arange(ymin,ymax+1)
        x = np.linspace(-1,1,num=xmax-xmin+1)
        y = np.linspace(-1,1,num=ymax-ymin+1)
        xx, yy = np.meshgrid(x,y)
        #xx_remap = 2*((xx - xmin) / (xmax) - 0.5)
        #yy_remap = 2*((yy - ymin) / (ymax) - 0.5)
        #rr = np.sqrt(xx_remap**2 + yy_remap**2)
        #pp = np.arctan2(yy_remap, xx_remap)
        rr = np.sqrt(xx**2 + yy**2)
        pp = np.arctan2(yy, xx)
        
        coeffs = (np.random.rand(j_max) - 0.5) * 2 #/ j_max
        Z = np.zeros_like(rr)
        for j,coeff in enumerate(coeffs):
            Z = Z + coeff * _Z_j(j, rr, pp)
            
        if circle:
            Z[rr>=1] = 0
    else:
        Z = Z_in
    if not type(data) is np.ndarray:
        Z = xp.array(Z)
    
    Zx, Zy = np.gradient(Z)
    mag = np.sqrt(Zx**2 + Zy**2).max()
    Zx = Zx * spread / mag
    Zy = Zy * spread / mag
    kxs = Zx[(data[0,:]-xmin).astype(int), (data[1,:]-ymin).astype(int)]
    kys = Zy[(data[0,:]-xmin).astype(int), (data[1,:]-ymin).astype(int)]
    num_events = kxs.size
    
    # multivariate skew norm in the direction of [kx, ky] adapted from:
    # https://gregorygundersen.com/blog/2020/12/29/multivariate-skew-normal/
    x = np.arange(-spread, spread+1)
    xx, yy = np.meshgrid(x, x)
    xy = np.dstack([xx,yy])
    xy = mvn._process_quantiles(xy, 2)
    L = np.diag([tightness,1-tightness])
    def kern(kx, ky):
        v1 = np.array([kx,ky]) #/ np.sqrt(kx**2 + ky**2)
        v2 = np.array([[0, -1],[1, 0]]) @ v1
        Q = np.hstack([v1[:,None],v2[:,None]])
        cov = Q @ L @ np.linalg.inv(Q) * np.sqrt(kx**2 + ky**2)**2
        a = v1
        aCa = a @ cov @ a
        delta = (1 / np.sqrt(1 + aCa)) * cov @ a
        cov_star = np.block([[np.ones(1), delta], [delta[:,None], cov]])
        x = mvn(np.zeros(3), cov_star).rvs(1)
        if x[0] <= 0: # for many samples consider vectorizing
            x[1] = -x[1]
            x[2] = -x[2]
        return x[1], x[2]
        #pdf = mvn(np.array([0,0]), cov).logpdf(xy)
        #cdf = norm(0,1).logcdf(np.dot(xy,a))
        #logpdf = np.log(2) + pdf + cdf
        #return np.exp(logpdf)
    
    data = data.copy()
    print('Generating offsets...')
    for event, (kx, ky) in enumerate(zip(kxs,kys)):
        if event % 1000 == 0:
            print(f'\tprogress: {event / num_events * 100:5.1f}%', end='\r')
        if kx==0 and ky==0:
            xoffset = yoffset = 0
        else:
            xoffset, yoffset = kern(float(kx), float(ky))
        data[0,event] = data[0,event] + int(xoffset)
        data[1,event] = data[1,event] + int(yoffset)
    print(f'\tprogress: {100:5.1f}%\nKernels generated!')
    
    if keep_in_beam:
        data[0, data[0,:]<xmin] = xmin + (xmin - data[0, data[0,:]<xmin])
        data[0, data[0,:]>xmax] = xmax - (data[0, data[0,:]>xmax] - xmax)
        data[1, data[1,:]<ymin] = ymin + (ymin - data[1, data[1,:]<ymin])
        data[1, data[1,:]>ymax] = ymax - (data[1, data[1,:]>ymax] - ymax)
    
    return data, Z

def _gen_hits(n_exp: float,
              n_bins: int,
              beams: list[Beam],
              toa_bounds: tuple[float,float],
              tot_bounds: tuple[float,float],
              circ: bool = True,
              verbose: bool = False,
              CUDA: bool = False) -> np.ndarray:
    if CUDA:
        gen = xp.random.default_rng()
    else:
        gen = np.random.default_rng()
    
    ## toa generator
    toa, number = _gen_toas(n_exp, n_bins, toa_bounds, gen, "normal", verbose, CUDA)
    
    ## positions generator
    if circ:
        pos = _gen_pos_circ(number, beams, gen, verbose, CUDA)
    else:
        pos = _gen_pos_rect(number, beams, gen, verbose, CUDA)
    
    ## tot generator
    tot_range = max(tot_bounds) - min(tot_bounds)
    
    tot = min(tot_bounds) + (gen.random(number) * tot_range)
    if verbose: print(f'tot written\n')
    
    # concatenate
    new_hits = np.concatenate([pos,
                               np.expand_dims(toa,axis=0),
                               np.expand_dims(tot,axis=0)],
                              axis=0)
    if verbose: print('concatenated new_hits together\n')
    
    return new_hits

def _gen_pairs(n_exp: float,
               n_bins: int,
               dToA_var: float,
               beams: list[Beam],
               toa_bounds: tuple[float,float],
               tot_bounds: tuple[float,float],
               linear_corr_strength: float,
               verbose: bool = False,
               CUDA: bool = False) -> np.ndarray:
    if CUDA:
        gen = xp.random.default_rng()
    else:
        gen = np.random.default_rng()
    
    ## toa generator
    idler_toas, number = _gen_toas(n_exp, n_bins, toa_bounds, gen, "super", verbose, CUDA)
    
    if verbose: print(f'\tgenerating dToAs...')
    
    # CuPy is behind on the eightball when it comes to the generator 
    # implementation, and so the functional approach must instead be called
    if CUDA:
        dToAs = xp.random.normal(0, dToA_var, number)
    else:
        dToAs = gen.normal(0, dToA_var, number)
        
    # / DT -> int -> * DT preserves discrete ToAs
    dToAs = (dToAs / DT).astype(int) * DT 
    signal_toas = idler_toas + dToAs
    
    # prevent the case where signal_toas.min() < toa_bounds.min(). This could 
    # shift the max toa to be past toa_bounds.max(), but that doesn't matter so
    # much as being below toa_bounds.min()
    sig_toa_diff = toa_bounds[0] - signal_toas.min()
    if sig_toa_diff > 0:
        idler_toas = sig_toa_diff + idler_toas
        signal_toas = sig_toa_diff + signal_toas

    if verbose: print(f'paired toas generated\n')
    
    ## positions generator
    idler_pos, signal_pos = _gen_pos_corr(number, linear_corr_strength, beams, gen,
                                       verbose, CUDA)
    
    ## find out-of-beam hits
    oob_idler = beams[0].in_beam(idler_pos)
    oob_signal = beams[1].in_beam(signal_pos)
    
    ## tot generator
    tot_range = max(tot_bounds) - min(tot_bounds)
    
    tot = min(tot_bounds) + (gen.random(number) * tot_range)
    if verbose: print(f'tot written\n')
    
    # concatenate while removing oob hits
    if verbose: print('concatenating all new hits together...\n')
    new_idler_hits = np.concatenate([idler_pos[:,oob_idler],
                                     np.expand_dims(idler_toas[oob_idler],axis=0),
                                     np.expand_dims(tot[oob_idler],axis=0)],
                                    axis=0)
    new_signal_hits = np.concatenate([signal_pos[:,oob_signal],
                                      np.expand_dims(signal_toas[oob_signal],axis=0),
                                      np.expand_dims(tot[oob_signal],axis=0)],
                                     axis=0)
    
    new_hits = np.concatenate([new_idler_hits, new_signal_hits], axis=1)
    
    return new_hits

def _gen_toas(n_exp: float, 
              n_bins: int, 
              toa_bounds: tuple[float, float],
              gen: np.random.Generator, 
              pois: Literal["normal", "super"] = "normal",
              verbose: bool = False,
              CUDA: bool = False):
    # sequential so that I can see progress as it takes a long time            
    if verbose: print(f'generating toas\n\t{n_bins=} {n_exp=}')

    if pois == "super":
        p_gen = lambda x,size: gen.geometric(x, size=size) - 1
        mu = 1 / (n_exp+1)
    elif pois == "normal":
        p_gen = gen.poisson
        mu = n_exp
    
    # this in-place generation should be more memory efficient
    if CUDA:
        toa_dist = xp.zeros(n_bins, dtype=int)
    else:
        toa_dist = np.zeros(n_bins, dtype=int)
    low_idx = 0
    number = 0
    for i in range(100):
        high_idx = low_idx + int(n_bins/100)
        curr = p_gen(mu, size=int(n_bins/100)).astype(int)
        number += int(np.sum(curr))
        toa_dist[low_idx:high_idx] = curr
        low_idx = high_idx
        if verbose: print(f'\t\t{i:3}% of toa generated',end='\r')
    curr = p_gen(mu,int(n_bins%100)).astype(int)
    number += int(np.sum(curr))
    toa_dist[low_idx:] = curr
    
    if verbose:
        print(f'\t\t{100:3}% of toa generated')
        print(f'\t{number} events generated')
        print(f'\ttoa dist generated')
        
    
    # this has some bug if I attempt to make times a cupy array at first. I am
    # getting around this by just doing it as numpy and then casting back to 
    # cupy when made
    #if CUDA: # see above
    #    times = (xp.arange(1,n_bins+1) * DT) + min(toa_bounds)
    #else:
    #    times = (np.arange(1,n_bins+1) * DT) + min(toa_bounds)
    #toa = np.repeat(times, toa_dist.tolist())
    times = (np.arange(1,n_bins+1) * DT) + min(toa_bounds)
    toa = np.repeat(times, asnumpy(toa_dist))
    
    if CUDA:
        toa = xp.array(toa)
    
    if verbose: print(f'toas generated')
    
    return toa, number

def _gen_pos_rect(number: int,
                  beams: list[Beam],
                  gen: np.random.Generator,
                  verbose:bool = False,
                  CUDA:bool = False) -> np.ndarray:
        
    pos = gen.random((len(beams), 2, int(np.ceil(number/len(beams)))))
    if verbose: print(f'position generator values made')
    
    for i,beam in enumerate(beams):
        spread_x = beam.right - beam.left
        spread_y = beam.top - beam.bottom
        pos[i,0,:] = (pos[i,0,:] * spread_x) + beam.left
        pos[i,1,:] = (pos[i,1,:] * spread_y) + beam.bottom
    if verbose: print(f'positions generated')
    
    # reshape positions and then cut off any excess positions past number
    pos = pos.reshape((2,-1))[:,:number]
    if CUDA:
        pos = asnumpy(pos.T) # annoying
        np.random.shuffle(pos.T)
        pos = xp.asarray(pos.T)
    else:
        gen.shuffle(pos,axis=1) # randomize order
    if verbose: print(f'positions reshaped and truncated\n')
    
    return pos

def _gen_pos_circ(number: int,
                  beams: list[Beam],
                  gen: np.random.Generator,
                  verbose:bool = False,
                  CUDA:bool = False) -> np.ndarray:
    centers = []
    width_x = []
    width_y = []
    for beam in beams:
        centers.append(beam.center)
        width_x.append((beam.right - beam.left) / 2)
        width_y.append((beam.top - beam.bottom) / 2)
        
    rands = gen.random((len(beams), 2, int(np.ceil(number/len(beams)))))
    if verbose: print(f'position generator values made')

    if CUDA:
        pos = xp.zeros((2,number))
    else:
        pos = np.zeros((2,number))
    if verbose: print(f'position output allocated')
    
    for i,beam in enumerate(beams):
        # high_idx causes excess hits (past number) to be removed from last beam
        low_idx = i * rands.shape[2]
        high_idx = min((i+1) * rands.shape[2], number)
        
        r = np.sqrt(rands[i,0,:])
        theta = rands[i,1,:] * 2 * np.pi 
        
        # surprisingly, this scaling method does actually preserve uniformity
        pos[0,low_idx:high_idx] = np.floor(centers[i][0] + \
            (r * np.cos(theta) * width_x[i]))[:(high_idx-low_idx)]
        pos[1,low_idx:high_idx] = np.floor(centers[i][1] + \
            (r * np.sin(theta) * width_y[i]))[:(high_idx-low_idx)]
    if CUDA:
        pos = asnumpy(pos.T) # annoying
        np.random.shuffle(pos.T)
        pos = xp.asarray(pos.T)
    else:
        gen.shuffle(pos,axis=1) # randomize order
    if verbose: print(f'positions generated')
    
    return pos

def _gen_pos_corr(number: int,
                  linear_corr_strength: float,
                  beams: list[Beam],
                  gen: np.random.Generator,
                  verbose:bool = False,
                  CUDA:bool = False) -> tuple[np.ndarray, np.ndarray]:
        
    # again, CuPy is behind on the eightball when it comes to the generator
    # implementation
    if CUDA:
        pos = xp.random.multivariate_normal(mean = (0,0),
                                    cov = np.array([[1, -linear_corr_strength],
                                                    [-linear_corr_strength, 1]]),
                                    size = (2, number))
    else:
        pos = gen.multivariate_normal(mean = (0,0),
                                    cov = np.array([[1, -linear_corr_strength],
                                                    [-linear_corr_strength, 1]]),
                                    size = (2, number))
    pos = np.transpose(pos, axes=[2,0,1]) # i,s; x,y; hits
    
    if verbose: print(f'position generator values made')
    
    if len(beams) == 1:
        beam = beams[0]
        center = beam.center
        width_x = (beam.right - beam.left) / 2
        width_y = (beam.top - beam.bottom) / 2
        
        # this may create some hits on the edges disproportionally, but it's 
        # hopefully minimal
        pos[:,0,:] = np.round((pos[:,0,:] / np.abs(pos[:,0,:].max()) \
                                      * width_x) + center[0])
                             #a_min = beam.left, 
                             #a_max = beam.right)
        pos[:,1,:] = np.round((pos[:,1,:] / np.abs(pos[:,1,:].max()) \
                                      * width_y) + center[1])
                             #a_min = beam.bottom, 
                             #a_max = beam.top)
    else:
        beam_i, beam_s = beams # will fail if len(beams)!=2
        
        center_i = beam_i.center
        center_s = beam_s.center
        width_xi = (beam_i.right - beam_i.left)# / 2
        width_xs = (beam_s.right - beam_s.left)# / 2
        width_yi = (beam_i.top - beam_i.bottom)# / 2
        width_ys = (beam_s.top - beam_s.bottom)# / 2
        
        # for debugging:
        #print(np.round((pos[0,0,:] / np.abs(pos[:,0,:].max()) * width_xi) + center_i[0]))
        #print(np.round((pos[1,0,:] / np.abs(pos[:,0,:].max()) * width_xs) + center_s[0]))
        #print(np.round((pos[0,1,:] / np.abs(pos[:,1,:].max()) * width_yi) + center_i[1]))
        #print(np.round((pos[1,1,:] / np.abs(pos[:,1,:].max()) * width_ys) + center_s[1]))
        
        xlimit = np.abs(pos[:,0,:].max())
        ylimit = np.abs(pos[:,0,:].max())
        
        pos[0,0,:] = np.round((pos[0,0,:] / xlimit * width_xi) + center_i[0])
        #                     beam_i.left,
        #                     beam_i.right)
        pos[1,0,:] = np.round((pos[1,0,:] / xlimit * width_xs) + center_s[0])
        #                     beam_s.left,
        #                     beam_s.right)
        pos[0,1,:] = np.round((pos[0,1,:] / ylimit * width_yi) + center_i[1])
        #                     beam_i.bottom,
        #                     beam_i.top)
        pos[1,1,:] = np.round((pos[1,1,:] / ylimit * width_ys) + center_s[1])
        #                     beam_s.bottom,
        #                     beam_s.top)
        
    if verbose: print(f'positions generated')
    
    idler_pos = pos[0,:,:]
    signal_pos = pos[1,:,:]
   
    if verbose: print(f'positions split')
    
    return idler_pos, signal_pos

def _Z_j(j: int, rho: float, phi: float):
    # Calculate Zernike polynomial. j is based on ANSI Standard Z80.28-210
    # n,m coeffs 1 to 15 based on j... could be analytic to go past 15?
    nm = np.array([(0,0),(1,-1),(1,1),(2,-2),(2,0),(2,2),(3,-3),(3,-1),(3,1),
                   (3,3),(4,-4),(4,-2),(4,0),(4,2),(4,4)])
    n = nm[j-1,0]
    m = nm[j-1,1]
    
    R = _R_nm(n, m, rho)
    Z = R * np.cos(m * phi) if m>=0 else R * np.sin(m * phi)
    
    ## normalize so that integral over unit circle = pi => var_{S^1}(Z) = 1
    norm = 2 * np.sqrt(np.pi) * np.sqrt(n+1) if m!=0 \
        else np.sqrt(np.pi) * np.sqrt(2) * np.sqrt(n+1)
    return norm * Z

def _R_nm(n:int, m:int, rho:float):
    # Calculate radial polynomials via Jacobi functions
    n, m = np.abs(n), np.abs(m)
    k = (n-m) / 2
    
    J = jacobi(k, m, 0)(1 - 2 * rho**2)
    R = (-1)**(k) * rho**m * J
    return R
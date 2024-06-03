'''
Contains simulation functions for making fake TPX3 or coincidence data. The 
majority of this submodule likely cannot be done in parallel as the typical 
sizes of added datat well exceed typical VRAM amounts.
'''

from tpx3_toolkit.core import Beam, DT, xp, asnumpy
from typing import Union
import numpy as np

def add_coherent(data: np.ndarray, 
                 num_hits: int, 
                 beams: list[Beam],
                 circular_beam: bool = True,
                 verbose: bool = False) -> tuple[np.ndarray, int]:
    '''
    Adds simulated hits to an existsing pix array, within the bounds of said 
    existing array.
    
    Parameters 
    ----------
    data: ndarray
        a pix array as described in t3.core.parse_raw_file()
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
    maxs = np.max(data,axis=1)
    mins = np.min(data,axis=1)
    
    toa_bounds = (maxs[2], mins[2])
    tot_bounds = (maxs[3], mins[3])
    
    n_bins = int((max(toa_bounds) - min(toa_bounds)) / DT)
    new_n_exp = num_hits / n_bins

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

    new_hits = _gen_hits(new_n_exp, 
                        n_bins,
                        beams, 
                        toa_bounds, 
                        tot_bounds,
                        circular_beam,
                        verbose,
                        CUDA)

    if verbose: print(f'\thits generated, concatenating...')

    if CUDA:
        extended_data = np.concatenate([data,xp.asarray(new_hits)], axis=1)
    else:
        extended_data = np.concatenate([asnumpy(data), new_hits])

    hits_added = new_hits.shape[1]
    
    if verbose: print(f'Done concatenaing! {hits_added} hits added')

    return extended_data, hits_added

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
    if data is None:
        assert time is not None, f'If creating a new data array, then the ' + \
                                 'time parameter must not be None'
        toa_bounds = (0, time*10**9)
    else:
        maxs = data.max(axis=1)
        mins = data.min(axis=1)
        
        toa_bounds = (mins[2], maxs[2])
        
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
                          linear_corr_strength,
                          verbose,
                          CUDA)
    
    if verbose: print(f'hits generated, concatenating to old data...')

    if data is not None:
        if CUDA:
            data_out_pre = np.concatenate([data,xp.asarray(new_hits)], axis=1)
        else:
            data_out_pre = np.concatenate([asnumpy(data), new_hits])
    else:
        data_out_pre = new_hits
            
    hits_added = new_hits.shape[1]
    
    if verbose: print(f'\nDone concatenaing! {hits_added} hits ({hits_added/2:.0f} pairs) added')
    
    # CUDA output if possible
    try:
        data_out = xp.array(data_out_pre)
    except:
        data_out = data_out_pre

    return data_out, hits_added
    
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
    toa, number = _gen_toas(n_exp, n_bins, toa_bounds, gen, verbose, CUDA)
    
    ## positions generator
    if circ:
        pos = _gen_pos_circ(number, beams, gen, verbose)
    else:
        pos = _gen_pos_rect(number, beams, gen, verbose)
    
    ## tot generator
    tot_range = max(tot_bounds) - min(tot_bounds)
    
    tot = min(tot_bounds) + (gen.random(number) * tot_range)
    if verbose: print(f'tot written\n')
    
    # concatenate
    new_hits = np.concatenate([pos,
                               np.expand_dims(toa,axis=0),
                               np.expand_dim(tot,axis=0)],
                              axis=0)
    if verbose: print('concatenated new_hits together\n')
    
    return new_hits

def _gen_pairs(n_exp: float,
               n_bins: int,
               dToA_var: float,
               beams: list[Beam],
               toa_bounds: tuple[float,float],
               linear_corr_strength: float,
               verbose: bool = False,
               CUDA: bool = False) -> np.ndarray:
    if CUDA:
        gen = xp.random.default_rng()
    else:
        gen = np.random.default_rng()
    
    ## toa generator
    idler_toas, number = _gen_toas(n_exp, n_bins, toa_bounds, gen, verbose, CUDA)
    
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
    if CUDA:
        idler_tots = xp.zeros(number)
        signal_tots = xp.copy(idler_tots)
    else:
        idler_tots = np.zeros(number)
        signal_tots = np.copy(idler_tots)
    if verbose: print(f'tot written\n')
    
    # concatenate while removing oob hits
    if verbose: print('concatenating all new hits together...\n')
    new_idler_hits = np.concatenate([idler_pos[:,oob_idler],
                                     np.expand_dims(idler_toas[oob_idler],axis=0),
                                     np.expand_dims(idler_tots[oob_idler],axis=0)],
                                    axis=0)
    new_signal_hits = np.concatenate([signal_pos[:,oob_signal],
                                      np.expand_dims(signal_toas[oob_signal],axis=0),
                                      np.expand_dims(signal_tots[oob_signal],axis=0)],
                                     axis=0)
    
    new_hits = np.concatenate([new_idler_hits, new_signal_hits], axis=1)
    
    return new_hits

def _gen_toas(n_exp: float, 
              n_bins: int, 
              toa_bounds: tuple[float, float],
              gen: np.random.Generator, 
              verbose: bool = False,
              CUDA: bool = False):
    # sequential so that I can see progress as it takes a long time
    if verbose: print(f'generating toas\n\t{n_bins=} {n_exp=}')
    
    toa_dist_parts = []
    number = 0
    for i in range(100):
        toa_dist_parts.append(np.floor(gen.poisson(n_exp,int(n_bins/100))))
        number += np.sum(toa_dist_parts[i])
        if verbose: print(f'\t\t{i:3}% of toa generated',end='\r')
    toa_dist_parts.append(np.floor(gen.poisson(n_exp,int(n_bins%100))))
    number += np.sum(toa_dist_parts[-1])
    number = int(number)
    
    if verbose:
        print(f'\t\t{100:3}% of toa generated')
        print(f'\t{number} pairs generated')
        print(f'\tconcatenating...')
        
    toa_dist = np.concatenate(toa_dist_parts).astype(int)
    if verbose: print(f'\ttoa dist generated')
    
    # this has some bug if I attempt to make times a cupy array at first. I am
    # getting around this by just doing it as numpy and then casting back to 
    # cupy when made
    #if CUDA: # see above
    #    times = (xp.arange(1,n_bins+1) * DT) + min(toa_bounds)
    #else:
    #    times = (np.arange(1,n_bins+1) * DT) + min(toa_bounds)
    #toa = np.repeat(times, asnumpy(toa_dist))
    times = (np.arange(1,n_bins+1) * DT) + min(toa_bounds)
    toa = np.repeat(times, asnumpy(toa_dist))
    
    if CUDA:
        toa = xp.array(toa)
    
    if verbose: print(f'toas generated')
    
    return toa, number

def _gen_pos_rect(number: int,
                  beams: list[Beam],
                  gen: np.random.Generator,
                  verbose:bool = False) -> np.ndarray:
        
    pos = gen.random((len(beams), 2, np.ceil(number/len(beams))))
    if verbose: print(f'position generator values made')
    
    for i,beam in enumerate(beams):
        spread_x = beam.right - beam.left
        spread_y = beam.top - beam.bottom
        pos[i,0,:] = (pos[i,0,:] * spread_x) + beam.left
        pos[i,1,:] = (pos[i,1,:] * spread_y) + beam.bottom
    if verbose: print(f'positions generated')
    
    # reshape positions and then cut off any excess positions past number
    pos = pos.reshape((2,-1))[:,:number]
    if verbose: print(f'positions reshaped and truncated\n')
    
    return pos

def _gen_pos_circ(number: int,
                  beams: list[Beam],
                  gen: np.random.Generator,
                  verbose:bool = False) -> np.ndarray:
    centers = []
    width_x = []
    width_y = []
    for beam in beams:
        centers.append(beam.center)
        width_x.append((beam.right - beam.left) / 2)
        width_y.append((beam.top - beam.bottom) / 2)
        
    rands = gen.random((len(beams), 2, np.ceil(number/len(beams))))
    if verbose: print(f'position generator values made')
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
            (r * np.cos(theta) * width_x))
        pos[1,low_idx:high_idx] = np.floor(centers[i][1] + \
            (r * np.sin(theta) * width_y)) 
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
'''
Contains simulation functions for making fake TPX3 or coincidence data. The 
majority of this submodule likely cannot be done in parallel as the typical 
sizes of added datat well exceed typical VRAM amounts.
'''

from tpx3_toolkit.core import Beam, DT, xp, asnumpy
import numpy as np

def add_hits(data: np.ndarray, 
             num: int, 
             beams: list[Beam],
             circular_beam: bool = True,
             verbose: bool = False) -> tuple[np.ndarray, int]:
    '''
    Adds simulated hits to an existsing pix array.
    
    Parameters 
    ----------
    data: ndarray
        a pix array as described in t3.core.parse_raw_file()
    num: int
        the target number of hits to add. This target will likely not be hit
        directly, but will be used to calculated the expectation value of hits
        in each temporal mode
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
    '''
    maxs = np.max(data,axis=1)
    mins = np.min(data,axis=1)
    
    toa_bounds = (maxs[2], mins[2])
    tot_bounds = (maxs[3], mins[3])
    
    n_bins = int((max(toa_bounds) - min(toa_bounds)) / DT)
    new_n_exp = num / n_bins

    try: # would prefer to change xp, but Python does not allow this, thus flag
        expected_size = n_bins * 64 # bytes (8 bits, 2x concat size, 4 fields)
        free_bytes = xp.cuda.Device(0).mem_info[1]
        if expected_size >= free_bytes:
            print(f'~{expected_size/(2**30):.2f}GiB of VRAM required, but only {free_bytes/(2**30):.2f}GiB availalbe. Using CPU.')
            CUDA = False
        else:
            CUDA = True
    except:
        CUDA = False
    
    if verbose: print(f'calculated n_exp = {new_n_exp:.4f}')

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
    toa, number = _gen_toas(n_exp, n_bins, toa_bounds, gen, verbose)
    
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
    if verbose: print('Concatenated new_hits together\n')
    
    return new_hits

def _gen_toas(n_exp: float, 
              n_bins: int, 
              toa_bounds: tuple[float, float],
              gen: np.random.Generator, 
              verbose: bool = False,
              CUDA: bool = False):
    # sequential so that I can see progress as it takes a long time
    if verbose: print(f'{n_bins=} {n_exp=}')
    
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
        print(f'\t{100:3}% of toa generated')
        print(f'\t{number} hits generated')
        print(f'\tconcatenating...\n')
        
    toa_dist = np.concatenate(toa_dist_parts).astype(int)
    if verbose: print(f'\n\ttoa dist generated')
    
    if CUDA: # see above
        times = (xp.arange(1,n_bins+1) * DT) + min(toa_bounds)
    else:
        times = (np.arange(1,n_bins+1) * DT) + min(toa_bounds)
    toa = np.repeat(times,toa_dist)
    if verbose: print(f'toas generated\n')
    
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
        centers.append(((beam.right + beam.left) / 2,
                        (beam.top + beam.bottom) / 2))
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
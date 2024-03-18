import numpy as np
from tpx3_toolkit.rust_tpx3 import *
from scipy.optimize import curve_fit

# CUDA parallelization attempt
try:
    import cupy as cp
    xp = cp
    asnumpy = cp.asnumpy
    print("tpx3_toolkit running in CUDA mode")
except: 
    xp = np
    axnumpy = np.asarray
    print("tpx3_toolkit runnning in CPU mode")

# global variables
DT = 1.5625 # ns, var for the time bin width

# classes
class Beam:
    '''
    Holds the position info for a beam.

    Parameters
    ----------
    left: int
        The pixel value of the left edge of the beam, inclusive
    bottom: int
        The pixel value of the bottom edge of the beam, inclusive
    right: int
        The pixel value of the right edge of the beam, inclusive
    top: int
        The pixel value of the top edge of the beam, inclusive
    '''
    def __init__(self,left:int,bottom:int,right:int,top:int):
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top

    @classmethod
    def fromString(cls,inp:str):
        inp = inp[1:-1] # remove '[' and ']'
        strings = inp.split(', ')
        return cls(int(strings[0]), int(strings[1]), 
                   int(strings[2]), int(strings[3]))

    def __str__(self):
        return f"[{self.left}, {self.bottom}, {self.right}, {self.top}]"

    def toList(self):
        return [self.left,self.bottom,self.right,self.top]
    
    @property
    def area(self):
        '''Area (in pixels) contained within the beam.'''
        x_spread = self.right - self.left
        y_spread = self.top - self.bottom
        return x_spread * y_spread
    
# Add classes which describes tdc, pix, and coincs. They should all be based
# around ndarrays, and just have documentation and getter attributes which make 
# the abstraction clearer and give access to named views. Optimally, I don't 
# want to lose the ability to call functions on the class which would work with
# ndarrays in general, but inheriting from ndarrays seems... not so great in 
# practice. It looks like writing a custom array container may be the move, and
# this is well describer here:
# https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch
# Another benefit is that I really don't need to implement most ndarray
# functionality since it's rare that operations are performed on whole arrays of
# these types. I will need to implement __array__(), __array_func__(), and
# __array_ufunc__() as outlined in the article above, as well as inhereting from
# NDArrayOperatorsMixin. Most of this is universal and should give most 
# functionality, except for __array_func__() which will need to cover specific
# NumPy functions I want to implement, like np.concatenate().

# wrappers for Rust functions
def parse_raw_file(inp_file: str) -> tuple[np.ndarray, np.ndarray]:
    ''' 
    Parses the information contained within a '.tpx3' raw data file.
    
    Parameters
    ----------
    inp_file: string
        a string which describes the path to the '.tpx3' raw data file which is
        to be parsed.

    Returns
    -------
    tdc: ndarray
        an array of the TDC data packets. Each entry is indexable by the
        following values:
            [0,:]: TriggerCounter (unitless)
                The number of times the trigger has been activated at the time
                of the TDC data acquisition.
            [1,:]: Timestamp (ns)
                The course time elapsed since the beginning of the data
                collection. Has a percision of 260 ps, and a maximum value of
                107.3741824 s.
    pix: ndarray
        an array of the Pixel data packets. Each entry is indexable by the
        following values:
            [0,:]: X (pixels)
                data column of the pixel address information.
            [1,:]: Y (pixels)
                data row of the pixel address information.
            [2,:]: ToA (ns)
                Time of arrival of particle. Percision of 1.5625 ns, maximum value
                of 26.853136 s.
            [3,:]: ToT (ns)
                Time of threshold. The amount of time it takes for the pixel to
                drop back below threshold value. Percision of 25 ns, maximum
                value of 25.575 us. 

    Notes
    -----
        Since the endianess of the data is dependant on the architecture of the 
    machine which the '.tpx3' file is created on, and also the default endianess
    of a C data type is dependant on the machine this code is being run on, this
    function compensates for this issue internally and returns standardized data
    orientation in the output.
        As such, all binary operations which must be performed on the data are
    done within this function. Further, any unit setting operations are also
    done within this file to get the proper units reported above for the output
    data types. That means that there will be a slight discrepency between the
    packet descriptions given in the Amsterdam Scientific Instruments SERVAL
    manual in Chapter 6: Appendix: file formats, and the dtypes output here.
    Most of this discrepency is in the `pixaddr` data field as it requires some
    binary processing to extract the x-y coordinates of the Pix data chunk. This
    is also evident in the processing of all the fine and course times
    components.
    '''    
    tdc,pix = _parse(inp_file) # type: ignore
    return (xp.asarray(tdc), xp.asarray(pix))

def chop_large_file(inp_file, 
                    max_size = 100):
    '''
    Chops a '.tpx3' file into smaller, properly formatted, '.tpx3' files.
    
    Parameters
    ----------
    inp_file: string
        The path to the file of interest
    max_size: float, optional, default=100
        The maximum size of the chopped files, in MB
        
    Notes
    -----
    If you are chopping the file '/path/to/file/big_file.tpx3', then this 
    function will create a folder '/path/to/file/big_file/' and place all of the
    chopped file parts in there with the names
    '/path/to/file/big_file/part_0.tpx3', '/path/to/file/big_file/part_1.tpx3',
    etc.. 
    If this folder already exists, it and all its contents will be deleted, and
    then a new folder with the new chopping will be created.
    '''
    _chop(inp_file, max_size) # type: ignore


# functions
def simplesort(arr,row):
    # will be either CuPy or NumPy depending on type of arr
    order = np.argsort(arr[row,:],kind='stable')
    return arr[:,order] 

def beam_mask(pix: np.ndarray,
              beamLocations: list[Beam],
              preserveSize: bool = False) -> np.ndarray:
    '''
    Masks the input array based on location of the beams.

    Parameters
    ----------
    pix: np.ndarray
        An array of pix values. See `parse_raw_file`.
    beamLocations: list(Beam)
        A list of beams (may only contain one beam). Each describes a
        rectangular range where data will be kept from the pix array.
    preserveSize: bool, optional, default = False
        When this parameter is `True`, the returned array is masked in such a
        way that it has the same size as the original array, and all entries
        which normally would be ignored in the masking, are instead set to zero.
    
    Returns
    -------
    out : np.ndarray
        Masked array of pix values based on the pixDataType.dt dtype. Masked
        entries are either removed (default, when preserveSize == `False`), or
        set to zero (when preservedSize == `True`).
    '''
    beamMasks = xp.full((len(beamLocations),pix.shape[1]),False)
    for beam,i in zip(beamLocations,range(len(beamLocations))):
        beamMasks[i] = np.logical_and(np.logical_and(np.logical_and(
            pix[1,:] <= beam.top, 
            pix[0,:] <= beam.right),
            pix[1,:] >= beam.bottom), 
            pix[0,:] >= beam.left) # ugly ANDing for CuPy consistency
    beamMask = np.any(beamMasks,axis=0)

    if preserveSize:
        out = pix.copy()
        out[:,beamMask] = 0
    else:
        out = pix[:,beamMask]
    
    return out

def clustering(pix: np.ndarray,
               timeWindow: float,
               spaceWindow: int,
               clusterRange: int=4,
               numScans: int=5) -> np.ndarray:
    '''
    Parameters
    ----------
    pix: np.ndarray
        An array of pix values. See `parse_raw_file`.
    timeWindow (ns): float
        The amount of time which defines the maximum time difference between two
        pix exntries at which point they will not be considered in the same
        cluster (thus not generated by the same source photon)
    spaceWindow (pixels): int
        The amount of space which defines the maximum time difference between
        two pix entries at which point they will not be considered in the same
        cluster (thus not generated by the same source photon)
    clusterRange: int, optional, default=4
        The maximum distance two entries can be in the array at which the check
        for space/time seperation occurs
    numScans: int, optional, default=5
        How many times to repeat this algorithm. Essentially works to increase the 
        clusterRange.

    Returns
    -------
    out: np.ndarray
        An array of pix values based on the `pixDataType.dt` dtype. These values
        now correspond to single photon events rather than the unclustered photon
        events as in the input array.
    '''
    
    pix = xp.asarray(pix)
    pix = simplesort(pix,2)
    pixprev = 0
    for scan in range(numScans):
        for offset in range(1,clusterRange):
            mask = xp.full(pix.shape[1], True)
            largerToT_mask = np.logical_not(mask)
            old_centroids = np.logical_not(mask)
            new_centroids = np.logical_not(mask)

            # Set mask to False wherever element is part of cluster, i.e. mask[i] 
        # being False indicates that it is in a cluster with mask[i-j]
            # (for i >=j). False elements in mask will be discarded
            mask[offset::offset] = np.invert(
                                    np.logical_and(
                                     np.diff(pix[2,::offset].astype(np.float64)) < timeWindow,
                                     np.sqrt(np.abs(np.diff(pix[0,::offset].astype(np.float64)))**2 
                                      + np.abs(np.diff(pix[1,::offset].astype(np.float64)))**2)  < spaceWindow
                                    )
                                   )

            # largerToT_mask[i] is True when ToT[i+j]-ToT[i] > 0. This array
            # is used to identify clusters where the centroid needs to be swapped.
            largerToT_mask[0:-offset:offset] = np.diff(pix[3,::offset]) > 0 

            # Identify indices of old centroids which have a element within
            # its cluster with a larger ToT. old_centroids[i] is True where 
            # mask[i] is True and largerToT_mask[i] is True and mask[i+j] is False
            old_centroids[0:-offset:offset] = np.logical_and(
                                               np.logical_and(
                                                largerToT_mask[0:-offset:offset],
                                                mask[0:-offset:offset]),
                                               np.invert(mask[offset::offset])
                                              )
            new_centroids[offset::offset] = old_centroids[0:-offset:offset]

            # Swap centroid to element with larger ToT
            mask[new_centroids] = True
            mask[old_centroids] = False

            # Throw away elments within identified clusters which are not centroids
            pix = pix[:,mask]

        if pixprev == pix.shape[1]: # convergence check
            break
        pixprev = pix.shape[1]
        
    # note that if CuPy was used, pix will be a CuPy array. I don't think this
    # is an issue, but we'll have to be careful here.
    return pix

def correct_ToT(pix: np.ndarray, 
                calibrationFile: str) -> np.ndarray:
    '''
    Performs the time of threshold (ToT) correction using a calibration file.
    
    Parameters
    ----------
    pix: np.ndarray
        An array of pix values. See `parse_raw_file`.
    calibrationFile: str
        A string pointing to the location of the ToT colibration file in the
        file system.
    
    Returns
    -------
    pix: np.ndarray
        An array of pix values. The ToA is corrected in this array based on the 
        ToT. See `parse_raw_file`. 

    Notes
    -----
        For the ToT correction, this fits an exponential function to the dToA vs
    ToT curve, then uses this fit to shift the data ToA based on the ToT.
    '''

    # must be done in NumPy to work with curve_fit. But that's ok, it's fast
    file = np.loadtxt(calibrationFile,skiprows=1)
    calibrationToT = file[:,0]
    calibrationdToA = file[:,1]

    f = lambda x,A,B: A*np.exp(-B*x)

    (popt,pcov) = curve_fit(f,calibrationToT,calibrationdToA,p0=[150,5e-3])

    out = pix.copy()
    out[2,:] = pix[2,:] - f(pix[3,:],popt[0],popt[1])

    return simplesort(out,2) # sorts array by ToA after the correction

def correct_ToA(pix: np.ndarray, 
                calibration_file: str):
    '''
    WORK IN PROGRESS
    Performs the time of arrival (ToA) correction using a calibration file.
    
    Parameters
    ----------
    pix: np.ndarray
        An array of pix values. See `parse_raw_file`.
    calibrationFile: str
        A string pointing to the location of the ToA colibration file in the
        file system.
    
    Returns
    -------
    pix: np.ndarray
        An array of pix values. The ToA is corrected in this array. See
        `parse_raw_file`. 

    Notes
    -----
        The correction to the ToA is done using a calibration file which has 
    offset times (in ns), which are defined for each pixel relative to the ToA
    of the (0,0) pixel. This file is generated by sending a short pulse of light
    to the TPX3CAM such that the whole sensor front is triggered at about the
    same time.
    '''
    # generate an array of offsets from the calibration file, then use that as 
    # as key for pix[0:1,:] to subtract those times from each arrived photon

def find_coincidences(pix: np.ndarray,
                      beams: list[list[Beam]],
                      coincidenceTimeWindow: float) -> np.ndarray:
    '''
    Finds all time coincidences between events in beamPix1 and beamPix2.

    Parameters
    ----------
    pix: np.ndarray
        An array of pix values. See `parse_raw_file`.
    beams: list[list[Beam]]
        A list of Beam objects which defines the locations where entries in the
        pix array will be tagged as coming from beam i. `len(beams)` must be 2+.
    coincidenceTimeWindow: float
        The amount of time which defines the maximum time difference between any
        i events between beam0, beam1, ..., beam(i-1), beam(i) which are
        considered coincidences.

    Returns
    -------
    coincidences: np.ndarray
        An array of paired pix values from beam0, beam1, ..., beam(i-1), beami,
        respectively, which are considred to be coincident with one another in
        time. Beami is referenced by the following:
            [i,:,:]:
                the beami info with each entry the same as in `pix` from
                `parse_raw_file`

    Notes
    -----
        The coincidences array dtype fields start from the zeroeth beam and count
    up, corresponding to the respective entries in the `beams` list. Thus, the
    coincidence entries corresponding to the beam defined in the zeroeth slot of
    the beams list would be indexed by `coincidences[0,:,:]`, while the
    entries corresponding to the beam defined in the first slot would be
    `coincidences[1,:,:]`, etc.
        To get the x-position of the photon from beam1 in coincidence 3 you
        would write `coincidences[1,0,3]`.
    '''
    assert len(beams)>1, f"Need len(beams) > 1, got: {len(beams)}"

    def replace_zeros_with_last_nonzeros(arr:np.ndarray) -> np.ndarray:
        # Generate an array corresponding to the indices in arr. Then everywhere
        # that arr["ToA"] is zero, also set the corresponding index to zero.
        # Then accumulate the maxmimum of the indices over the indices array.
        # This will look like the following: 
        #           arr["ToA"] = [0   ,0   ,4e10,0   ,5e10,0   ,0   ,8e10,9e10]
        #                  idx = [0   ,1   ,2   ,3   ,4   ,5   ,6   ,7   ,8   ]
        #                  (1)-> [0   ,0   ,2   ,0   ,4   ,0   ,0   ,7   ,8   ]
        #                  (2)-> [0   ,0   ,2   ,2   ,4   ,4   ,4   ,7   ,8   ]
        #   => arr["ToA"][idx] = [0   ,0   ,4e10,4e10,5e10,5e10,5e10,8e10,9e10]
        # This needs to be done in NumPy since this process cannot be 
        # realistically parallelized in any beneficial way
        idx = np.arange(arr.shape[1])
        idx[asnumpy(arr[2,:] == 0)] = 0 # (1)
        idx = np.maximum.accumulate(idx) # (2)
        return arr[:,xp.asarray(idx)] # possible CuPy conversion for consistency

    pix = simplesort(pix,2) # Make sure the array is first sorted!

    coincidences = xp.zeros((len(beams),4,pix.shape[1]))
    for beam,i in zip(beams,range(len(beams))):
        # Fill all entries not within the current beam with 0, then replace each
        # zero entry with the last prior non-zero entry.
        curr = beam_mask(pix,beam,preserveSize=True)
        coincidences[i,:,:] = replace_zeros_with_last_nonzeros(curr)
    dT = np.amax(coincidences[:,2,:],axis=0) - np.amin(coincidences[:,2,:],axis=0)
    keepIndices = np.where(dT<coincidenceTimeWindow)[0]

    return coincidences[:,:,keepIndices]

def generate_ToA_correction():
    '''
    Placeholder for now
    '''

def process_Coincidences(inpFile: str,
                         calibrationFile: str,
                         beamSs: list[Beam],
                         beamIs: list[Beam],
                         timeWindow: float,
                         spaceWindow: int,
                         coincidenceTimeWindow: float,
                         clusterRange: int=0,
                         numScans: int=0) -> np.ndarray:
    '''
    Processes a raw tpx3 file to find coincidences between a set of signal and
    idler beams.
    
    Parameters
    ----------
    inpFile: string
        a string which describes the path to the '.tpx3' raw data file which is
        to be parsed.
    calibrationFile: str
        A string pointing to the location of the ToA colibration file in the
        file system. This file is generated by the `generate_ToA_correction`
        function.
    beamSs: list[Beam]
        A list of Beam objects describing the bounding box(es) of the signal
        beam(s) on the camera pixel array.
    beamIs: list[Beam]
        A list of Beam objects describing the bounding box(es) of the idler 
        beam(s) on the camera pixel array.
    timeWindow (ns): float
        The amount of time which defines the maximum time difference between two
        pix exntries at which point they will not be considered in the same
        cluster (thus not generated by the same source photon)
    spaceWindow (pixels): int
        The amount of space which defines the maximum time difference between
        two pix entries at which point they will not be considered in the same
        cluster (thus not generated by the same source photon)
    coincidenceTimeWindow: float
        The amount of time which defines the maximum time difference between any
        two events between beam1 and beam2 which are considered coincidences.
    clusterRange: int, optional, default=4
        The maximum distance two entries can be in the array at which the check
        for space/time seperation occurs
    numScans: int, optional, default=5
        How many times to repeat this algorithm. Essentially works to increase the 
        clusterRange.

    Returns
    -------
    coincidences: np.ndarray
        An array of paired pix values from the signal beam(s) and idler beam(s),
        respectively, which are considred to be coincident with one another in
        time. The beams are referenced by the following:
            [0,:,:]:
                the idler beam(s) info with each entry the same as in `pix` from
                `parse_raw_file`
            [1,:,:]:
                the signal beam(s) info with each entry the same as in `pix`
                from `parse_raw_file`

    Notes
    -----
    To get the x-position of the photon from beam1 in coincidence 3 you would
    write `coincidences[1,0,3]`.
    '''

    (tdc,pix) = parse_raw_file(inpFile)

    pix = beam_mask(pix,[*beamSs,*beamIs])

    if pix.shape[1] == 0:
        print("No events in defined beam locations")
        return np.array([])

    if(clusterRange == 0 and numScans == 0):
        pix = clustering(pix,timeWindow,spaceWindow)
    else:
        pix = clustering(pix,timeWindow,spaceWindow,clusterRange,numScans)

    pix = correct_ToT(pix,calibrationFile)

    coincidences = find_coincidences(pix,[beamSs,beamIs],coincidenceTimeWindow)
    return coincidences
    
if __name__ == '__main__':
    import os
    try:
        import functiontrace #type: ignore
    except:
        pass
    
    inpFile = os.path.dirname(os.path.realpath(__file__)) + \
        r'/examples/demo_file.tpx3'
    calibFile = os.path.dirname(os.path.realpath(__file__)) + \
        r'/examples/TOT correction curve new firmware GST.txt'
        
    try:
        functiontrace.trace()
    except:
        pass
    
    test = process_Coincidences(inpFile, 
                                calibFile, 
                                beamSs = [Beam(65, 57, 130, 188)], 
                                beamIs = [Beam(130, 57, 195, 188)],
                                timeWindow = 250, 
                                spaceWindow = 20, 
                                coincidenceTimeWindow = 1000, 
                                clusterRange = 30, 
                                numScans = 20)
    print(test)
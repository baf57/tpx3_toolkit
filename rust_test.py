import rust_parse
import tpx3_toolkit as t3
#import numpy as np
#import time

inpFile = '/home/brayden/Programs/my_git_dirs/tpx3_toolkit/tpx3_toolkit/examples/demo_file.tpx3'

#t0r = time.time()
(tdc, pix) = rust_parse.parse(inpFile)
#tdc = np.array(tdc).transpose()
#pix = np.array(pix).transpose()
#t1r = time.time()

#t0p = time.time()
(tdc_0, pix_0) = t3.parse_raw_file(inpFile)
#t1p = time.time()

print(f"# of Rust TDC entries: {len(tdc)}")
#print(f"Rust time: {t1r-t0r}")
#print(f"NmPy time: {t1p-t0p}")
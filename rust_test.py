import rust_parse
import tpx3_toolkit as t3
import numpy as np
import time

#inpFile = '/home/brayden/Documents/Education/Graduate/Lab/Quantum Imaging/Data/04-25-2023/momentum_000013_Optimal.tpx3'
inpFile = '/home/brayden/Programs/my_git_dirs/tpx3_toolkit/tpx3_toolkit/examples/demo_file.tpx3'

size = 10**4

t0r = time.time()
(tdc, pix) = rust_parse.parse(inpFile)
#tdc = np.array(tdc).transpose()
#pix = np.array(pix).transpose()
#outi = rust_parse.alloc_test_in(size)
t1r = time.time()

t0p = time.time()
(tdc_0, pix_0) = t3.parse_raw_file(inpFile)
#outo = np.zeros((size,size))
#rust_parse.alloc_test_out(outo)
t1p = time.time()

#t0c = time.time()
#outc = rust_parse.alloc_test_combo(size)
#outc = np.array(outc)
#t1c = time.time()

#print(f"# of Rust TDC entries: {len(tdc)}")
#print(f"Arrays equal: {np.all(np.equal(tdc,tdc_0))}, {np.all(np.equal(pix,pix_0))}")
print(f"pix: {pix[:,0]}")#, tdc: {tdc[:,0]}")
print(f"pix_0: {pix_0[:,0]}")#, tdc_0: {tdc_0[:,0]}")
print(f"Rust time: {t1r-t0r}")
print(f"NmPy time: {t1p-t0p}")
#print(f"Combo time: {t1c-t0c}")
print(f"Ratio: {(t1r-t0r) / (t1p-t0p)}")
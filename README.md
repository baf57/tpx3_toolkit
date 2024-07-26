# tpx3_toolkit
A Python toolkit for loading, parsing, and analyzing TimePix3 generated flies 
(.tpx3), as well as visualizing and manipulating data from these files.

# Installation
Installation instructions can be found in the 
[wiki](https://github.com/baf57/tpx3_toolkit/wiki/Installation-and-Updating), 
and following them is strongly recommended.

# Dependencies
The following Python packages are required to use this toolkit. They should be 
automatically installed when this toolkit is installed when using `pip`
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)
* [matplotlib](https://matplotlib.org/)

The following Python package is optional if you decide you can/want to utilize
parallel processing with CUDA. You may choose to install it with `pip` 
separately from the toolkit installation
* [cupy](https://cupy.dev/)

Along with this, the rust compiler is required to be present on the system to 
compile the rust components. Detailed instructions on how to acquire this for
your specific OS can be found [here](https://www.rust-lang.org/tools/install).
Lastly, [maturin](https://github.com/PyO3/maturin) may be needed if any rust 
code edits are performed. This can be installed using `cargo` once the rust
compiler is installed.

use backend::i_parse;
use pyo3::prelude::*;
pub mod backend;

#[pyfunction]
fn parse(inp_file: &str) -> PyResult<(Vec<[f64;2]>, Vec<[f64;4]>)> {
    let tdc: Vec<[f64;2]>;
    let pix: Vec<[f64;4]>;

    (tdc,pix) = i_parse(inp_file)?;
    
    Ok((tdc,pix))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_parse(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    Ok(())
}

/* How to use

CLI:
% maturin develop

Run the `rust_test.py` file in the root directory to test
*/
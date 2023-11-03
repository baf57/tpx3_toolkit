use backend::i_parse;
use numpy::PyArray2;
use pyo3::prelude::*;
pub mod backend;


/// A Python module implemented in Rust.
#[pymodule]
fn rust_parse(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn _parse<'py>(py:Python<'py>, inp_file: &str) -> PyResult<(&'py PyArray2<f64>,&'py PyArray2<f64>)> {

        let (tdc,pix) = i_parse(inp_file)?;
        
        Ok((PyArray2::from_vec2(py, &tdc).unwrap(), PyArray2::from_vec2(py, &pix).unwrap()))
    }

    Ok(())
}

/* How to compile
CLI:
% maturin develop
*/
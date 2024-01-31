use parse::i_parse;
use chop::chop;
use numpy::PyArray2;
use pyo3::prelude::*;
pub mod parse;
pub mod chop;


/// A Python module implemented in Rust.
#[pymodule]
fn rust_tpx3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn _parse<'py>(py:Python<'py>, inp_file: &str) -> PyResult<(&'py PyArray2<f64>,&'py PyArray2<f64>)> {

        let (tdc,pix) = i_parse(inp_file)?;
        
        Ok((PyArray2::from_vec2(py, &tdc).unwrap(), PyArray2::from_vec2(py, &pix).unwrap()))
    }

    #[pyfn(m)]
    fn _chop<'py>(_py:Python<'py>, inp_file: &str, max_size: f64) -> PyResult<()>{
        chop(inp_file, max_size)?;
        Ok(())
    }

    Ok(())
}

/* How to compile
CLI:
% maturin develop
*/
use backend::i_parse;
use numpy::PyArray2;
use pyo3::prelude::*;
pub mod backend;


/// A Python module implemented in Rust.
#[pymodule]
fn rust_parse(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn parse<'py>(py:Python<'py>, inp_file: &str) -> PyResult<(&'py PyArray2<f64>,&'py PyArray2<f64>)> {

        let (tdc,pix) = i_parse(inp_file)?;
        
        Ok((PyArray2::from_vec2(py, &tdc).unwrap(), PyArray2::from_vec2(py, &pix).unwrap()))
    }

//    #[pyfn(m)]
//    fn alloc_test_in<'py>(py:Python<'py>, size: usize) -> PyResult<&PyArray2<f64>>{ // memory managed by Python so safe
//        //let out: Vec<f64> = vec![0.0; size];
//
//       //Ok(out.to_pyarray(py))
//       Ok(PyArray2::<f64>::zeros(py, [size, size], false))
//    }
//
//    #[pyfn(m)]
//    fn alloc_test_combo(size: usize) -> PyResult<Vec<Vec<f64>>>{ // memory managed by Python so safe
//        let out: Vec<Vec<f64>> = vec![vec![0.0; size]; size];
//
//       Ok(out)
//    }
//
//    #[pyfn(m)]
//    fn alloc_test_out(_val: &PyArray2<f64>) -> PyResult<()>{
//        Ok(())
//    }

    Ok(())
}

/* How to use

CLI:
% maturin develop

Run the `rust_test.py` file in the root directory to test
*/
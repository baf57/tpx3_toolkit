use std::fs::File;
use std::io::Read;
use pyo3::prelude::*;

#[pyfunction]
fn parse(inp_file: &str) -> PyResult<(Vec<[i8; 2]>, Vec<[i8; 4]>)> {
    /* The idea is to read the file line by line, and on each line which is of
    one of the two correct types, to process the data. 
    Another idea is to sort the data as it is read from the file stream, which
    may cut down on time, but this is a future problem, and I will just
    reimplement the Python and NumPy code first. */

    let mut tdc: Vec<[i8;2]> = vec![];
    let mut pix: Vec<[i8;4]> = vec![];

    // open file with buffer. If the file open fails then send that up
    let mut f: File = File::open(inp_file)?;
    let mut buffer: Vec<u8> = Vec::new();

    f.read_to_end(&mut buffer)?;

    /* iterate over all the bytes. If the current byte is the end of a word, 
    then check for the byte type. If it is 0x6 then parse like TDC, if it is 0xB
    then parse like pix. Otherwise ignore it. */
    for (i, byte) in buffer.iter().enumerate(){
        println!("Byte {i} is {byte:08b}");
        if i == 7 {
            println!("{:?}", &buffer[..i+1]);
            break;
        }
    }

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
% cd src
% maturin develop

Current Python code to test this with:
import rust_parse

inpFile = '/home/brayden/Programs/my_git_dirs/tpx3_toolkit/tpx3_toolkit/examples/demo_file.tpx3'

size = rust_parse.parse(inpFile)
 
*/
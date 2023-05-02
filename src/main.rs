use backend::i_parse;
pub mod backend;
/* This is a testing environment for doing timing analysis of the rust code
   from within the rust environment to see if it is the transfer of data from
   python -> rust that is causing the slowdown, or if it is the issue of the 
   rust code itself. */

fn main() {
    let tdc: Vec<[f64;2]>; 
    let _pix: Vec<[f64;4]>;
    let inp_file = "/home/brayden/Programs/my_git_dirs/tpx3_toolkit/tpx3_toolkit/examples/demo_file.tpx3";

    (tdc,_pix) = i_parse(inp_file).expect("Parse failed!");
    println!("Finished reading file! TDC has {} entries", tdc.len());
}
use backend::i_parse;
pub mod backend;

fn main(){
    let tdc: Vec<[f64;2]>;
    let _pix: Vec<[f64;4]>;
    let inp_file = "/home/brayden/Documents/Education/Graduate/Lab/Quantum Imaging/Data/04-25-2023/momentum_000013_Optimal.tpx3";
    
    (tdc,_pix) = i_parse(inp_file).expect("Error parsing file: ");

    println!("# of TDC entries: {}", tdc.len())
}
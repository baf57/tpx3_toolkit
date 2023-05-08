use backend::i_parse;
pub mod backend;

fn main(){
    let tdc: Vec<Vec<f64>>;
    let pix: Vec<Vec<f64>>;
    //let inp_file = "/home/brayden/Documents/Education/Graduate/Lab/Quantum Imaging/Data/04-25-2023/momentum_000013_Optimal.tpx3";
    let inp_file = "/home/brayden/Programs/my_git_dirs/tpx3_toolkit/tpx3_toolkit/examples/demo_file.tpx3";

    (tdc,pix) = i_parse(inp_file).expect("Error parsing file: ");

    println!("# of TDC entries: {}", tdc[0].len());
    println!("# of Pix entries: {}", pix[0].len());
}
use std::fs::File;
//use std::time::Instant;
use std::io::Read;

pub fn i_parse(inp_file: &str) 
                        -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>),std::io::Error> {
    /* The idea is to read the file line by line, and on each line which is of
    one of the two correct types, to process the data. 
    Another idea is to sort the data as it is read from the file stream, which
    may cut down on time, but this is a future problem, and I will just
    reimplement the Python and NumPy code first. */
    //let now = Instant::now();

    let mut tdc: Vec<Vec<f64>> = vec![vec![],vec![]];
    let mut pix: Vec<Vec<f64>> = vec![vec![],vec![],vec![],vec![]];

    // open file with buffer. If the file open fails then send that up
    let mut f: File = File::open(inp_file)?;
    let mut buffer: Vec<u8> = Vec::new();

    f.read_to_end(&mut buffer)?; // read file to buffer

    let tdc_byte: u8 = 0x60;
    let pix_byte: u8 = 0xB0; 

    let mut tdc_buffer: u64; 
    let mut trigger_counter: u16;
    let mut time_stamp: u32;
    let mut stamp: u8;

    let mut pix_buffer: u64;
    let mut d_col: u8;
    let mut s_pix: u8;
    let mut pix_raw: u8;
    let mut t_o_a: u16;
    let mut t_o_t: u16;
    let mut f_t_o_a: u16;
    let mut spidr_time: u16;

    /* iterate over all the bytes. If the current byte is the end of a word, 
    then check for the byte type. If it is 0x6 then parse like TDC, if it is 0xB
    then parse like pix. Otherwise ignore it. */
    for (i, byte) in buffer.iter().enumerate(){
        if i%8 == 7 { // end of word
            if (byte&0b11110000) == tdc_byte{
                tdc_buffer = 0;
                for (j, ent) in buffer[i-7..i+1].iter().enumerate(){ // recreate word
                    tdc_buffer += (*ent as u64) << (8 * j);
                }
                // note that "as" clips the value
                // these masks below are all wonky, but it's what the devs use
                trigger_counter = ((tdc_buffer>>44) & 0xFFF) as u16;
                time_stamp = ((tdc_buffer>>9) & 0x7FFFFFFFF) as u32;
                stamp = ((tdc_buffer>>5) & 0xF) as u8;
                
                tdc[0].push(trigger_counter as f64);
                tdc[1].push((stamp as f64 * 260e-3) +
                            (time_stamp as f64 * 3125e-3)); // seems wrong? but works?
            }
            else if (byte&0b11110000) == pix_byte{
                pix_buffer = 0;
                for (j,ent) in buffer[i-7..i+1].iter().enumerate(){
                    pix_buffer += (*ent as u64) << (8 * j);
                }
                d_col = ((pix_buffer>>53) & 0x7F) as u8;
                s_pix = ((pix_buffer>>47) & 0x3F) as u8; 
                pix_raw = ((pix_buffer>>44) & 0x7) as u8;
                t_o_a = ((pix_buffer>>30) & 0x3FFF) as u16;
                t_o_t = ((pix_buffer>>20) & 0x3FF) as u16;
                f_t_o_a = ((pix_buffer>>16) & 0xF) as u16;
                spidr_time = (pix_buffer & 0xFFFF) as u16;
                //let tftoa = (t_o_a<<4) | (!f_t_o_a & 0xF);
                //println!("rust:");
                //println!("\ttoa: {t_o_a}");
                //println!("\ttot: {t_o_t}");
                //println!("\tftoa: {tftoa:04b}");
                //println!("\tspidr: {spidr_time}");
                
                // Axis need to be mirrored to reflect the actual camera orientation in lab frame
                pix[0].push(255.0 - (((s_pix<<2) as f64) + ((pix_raw & 0x3) as f64)));
                pix[1].push(255.0 - (((d_col<<1) as f64) + ((pix_raw/4) as f64)));
                pix[2].push(((spidr_time as f64) * 25.0 * 16384.0) + 
                            ((((t_o_a<<4) | (!f_t_o_a & 0xF)) as f64) * (25.0/16.0))); //wrong
                pix[3].push((t_o_t as f64) * 25.0);

                //println!("r: {i:x}: {pix_buffer:064b}");
                //break;
            }
        }
    }

    //let elapsed = now.elapsed();
    //println!("Time inside Rust for i_parse: {:.2?}", elapsed);
    //println!("Size of pix: {}, of tdc: {}", pix[0].len(), tdc[0].len());

    Ok((tdc,pix))
}
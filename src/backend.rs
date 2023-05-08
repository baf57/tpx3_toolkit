use std::fs::File;
use std::time::Instant;
use std::io::Read;

pub fn i_parse(inp_file: &str) 
                        -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>),std::io::Error> {
    /* The idea is to read the file line by line, and on each line which is of
    one of the two correct types, to process the data. 
    Another idea is to sort the data as it is read from the file stream, which
    may cut down on time, but this is a future problem, and I will just
    reimplement the Python and NumPy code first. */
    let now = Instant::now();

    let mut tdc: Vec<Vec<f64>> = vec![vec![],vec![]];
    let mut pix: Vec<Vec<f64>> = vec![vec![],vec![],vec![],vec![]];

    // open file with buffer. If the file open fails then send that up
    let mut f: File = File::open(inp_file)?;
    let mut buffer: Vec<u8> = Vec::new();

    f.read_to_end(&mut buffer)?; // read file to buffer

    let tdc_byte: u8 = 0x6;
    let pix_byte: u8 = 0xB; 

    let mut tdc_buffer: u64; 
    let mut trigger_counter: u16;
    let mut time_stamp: u32;
    let mut stamp: u8;

    let mut pix_buffer: u64; 
    let mut dcol: u8;
    let mut spix: u8;
    let mut pix_raw: u8;
    let mut toA: u16;
    let mut toT: u16;
    let mut fToA: u16;
    let mut spidr_time: u16;

    /* iterate over all the bytes. If the current byte is the end of a word, 
    then check for the byte type. If it is 0x6 then parse like TDC, if it is 0xB
    then parse like pix. Otherwise ignore it. */
    for (i, byte) in buffer.iter().enumerate(){
        if i%8 == 7 { // end of word
            if (byte&0b00001111) == tdc_byte{
                tdc_buffer = 0;
                for ent in buffer[i-7..i+1].iter(){
                    tdc_buffer <<= 8;
                    tdc_buffer += *ent as u64;
                }
                // note that as clips the value
                // these masks below are all wonky, but it's what the devs use
                trigger_counter = ((tdc_buffer>>44) & 0xFFF) as u16;
                time_stamp = ((tdc_buffer>>9) & 0x7FFFFFFFF) as u32;
                stamp = ((tdc_buffer>>5) & 0xF) as u8;
                
                tdc[0].push(trigger_counter as f64);
                tdc[1].push((stamp as f64 * 260e-3) +
                            (time_stamp as f64 * 3125e-3));
            }
            else if (byte&0b00001111) == pix_byte{
                pix_buffer = 0;
                for ent in buffer[i-7..i+1].iter(){
                    pix_buffer <<= 8;
                    pix_buffer += *ent as u64;
                }
                dcol = ((pix_buffer>>53) & 0x7F) as u8;
                spix = ((pix_buffer>>47) & 0x3F) as u8; 
                pix_raw = ((pix_buffer>>44) & 0x7) as u8;
                toA = ((pix_buffer>>30) & 0x3FFF) as u16;
                toT = ((pix_buffer>>20) & 0x3FF) as u16;
                fToA = ((pix_buffer>>16) & 0xF) as u16;
                spidr_time = (pix_buffer & 0xFFFF) as u16;

                pix[0].push(((dcol<<1) as f64) + ((pix_raw/4) as f64));
                pix[1].push(((spix<<2) as f64) + ((pix_raw & 0xF) as f64));
                pix[2].push(((spidr_time as f64) * 25.0 * 16384.0) + 
                            ((((toA<<4) | (fToA & 0xF)) as f64) * (25.0/16.0)));
                pix[3].push((toT as f64) * 25.0)
            }
        }
    }

    let elapsed = now.elapsed();
    println!("Time inside Rust for i_parse: {:.2?}", elapsed);

    Ok((tdc,pix))
}
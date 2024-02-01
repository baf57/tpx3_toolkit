use std::fs::File;
use std::io::{self, BufRead, BufReader, ErrorKind, Read, Write};

struct Storage{
    storage: Vec<u8>,
    max_size: u64,
}

impl Storage {
    pub fn new(max_size: u64) -> Storage{
        Storage { 
            storage: Vec::new(),
            max_size: max_size,
        }
    }

    pub fn store(&mut self, mut buffer: Vec<u8>) -> io::Result<()>{
        if self.storage.len() + buffer.len() <= self.max_size.try_into().unwrap(){
            self.storage.append(&mut buffer);
            return Ok(());
        }
        Err(ErrorKind::InvalidData.into())
    }

    pub fn write(&mut self, out_file: &str) -> io::Result<()>{
        let mut f = File::create(out_file)?;
        f.write_all(&self.storage)?;
        self.storage = Vec::new();
        Ok(())
    }
}

const BUFFERSIZE: usize = 1024*32; // 32 KiB buffer

pub fn chop(inp_file: &str, max_size: f64) -> io::Result<()> {
    // Just read the file byte-by-byte into a vec. Once the sequence {b'T',
    // b'P', b'X'} is seen, try to put the buffer into storage.  If len(buffer)
    // + len(storage) > max_size_bytes, then write storage to a file, then flush
    // it and write buffer to storage. Otherwise add it to storage.  Continue
    // this until the end of the file is reached, attempt to add the final
    // buffer to the storage, then write the storage and finish.
    let mut buffer: Vec<u8> = Vec::new();
    let mut curr_byte: u8;

    let f = File::open(inp_file)?;
    let mut reader = BufReader::with_capacity(BUFFERSIZE, &f);

    // max_size is in MB
    let max_size_bytes: u64 = (max_size * f64::powf(10.0,6.0)) as u64;    
    let mut storage: Storage = Storage::new(max_size_bytes);

    let folder_name = &inp_file[..inp_file.len()-5];
    let name_start = format!("{}/part_", folder_name);

    // if folder exists, delete it and all its contents, then make a new one
    match std::fs::create_dir(folder_name){
        Ok(()) => (),
        Err(_error) => {
            std::fs::remove_dir_all(folder_name)?;
            std::fs::create_dir(folder_name)?;
        }
    };

    // actual writing of data to files
    let mut counter: u8 = 0;
    if f.metadata()?.len() > max_size_bytes {
        loop{
            let real_buffer = reader.fill_buf()?;
            let buffer_length = real_buffer.len();

            // EOF
            if buffer_length == 0{
                break;
            }

            for byte in real_buffer.bytes(){
                // error handling
                match byte{
                    Ok(byte) => {curr_byte = byte}
                    Err(err) => panic!("Read error: {:?}", err)
                }
                buffer.push(curr_byte);

                // skip first TPX, but check for second one to write to storage
                if buffer.len() > 4 &&
                buffer[(buffer.len()-3)..] == vec![b'T',b'P',b'X'] {
                    match storage.store(buffer[..(buffer.len()-3)].to_vec()){
                        Ok(()) => (),
                        Err(error) => match error.kind(){
                            ErrorKind::InvalidData => {
                                storage.write(&format!("{}{:03}.tpx3",
                                                    name_start, counter))?;
                                counter+= 1;
                                storage.store(buffer[..(buffer.len()-3)].to_vec())?;
                            },
                            other_error => panic!("Unknown storage error: {:?}", 
                                                other_error)
                        }
                    }

                    // properly start next buffer
                    buffer = Vec::new();
                    buffer.push(b'T');
                    buffer.push(b'P');
                    buffer.push(b'X');
                }
            }

            // Don't read the same data twice
            reader.consume(buffer_length);
        }

        //println!("Buffer size: {:?}", (buffer.len()*8));
        //println!("Counter: {:?}", counter);

        // final check to get everything left in buffer into storage
        match storage.store(buffer.clone()){
            Ok(()) => (),
            Err(error) => match error.kind(){
                ErrorKind::InvalidData => {
                    storage.write(&format!("{}{:03}.tpx3",
                                            name_start, counter))?;
                    counter+= 1;
                    storage.store(buffer.clone())?;
                },
                other_error => panic!("Unknown storage error: {:?}", 
                                        other_error)
            }
        }
        storage.write(&format!("{}{:03}.tpx3", name_start, counter))?;
    }

    Ok(())
}

#[test]
fn parse_test() -> io::Result<()>{
    chop("test/5s_noise_static_000007.tpx3",500.0)
}
use std::{io, num};

/// The Error type
#[derive(Debug)]
pub enum Error {
    /// I/O errors
    Io(io::Error),
    /// Parse Int error
    ParseInt(num::ParseIntError),
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<num::ParseIntError> for Error {
    fn from(err: num::ParseIntError) -> Self {
        Self::ParseInt(err)
    }
}

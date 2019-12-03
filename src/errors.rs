use std::{error, fmt, io};

/// The Error type
#[derive(Debug)]
pub enum Error {
    /// I/O errors
    Io(io::Error),
    /// Invalid entry in dictionary
    InvalidDictEntry(String),
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Error::Io(ref err) => err.fmt(f),
            Error::InvalidDictEntry(ref err) => write!(f, "invalid dictionary entry: {}", err),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Error::Io(ref err) => Some(err),
            Error::InvalidDictEntry(_) => None,
        }
    }
}

//! Bitstream I/O: MSB-first reader/writer and Exp-Golomb codes.
//!
//! Ports `reference/codec/decoder/core/inc/{bit_stream,dec_golomb}.h`. The C
//! reader keeps a 32-bit cache refilled 16 bits at a time; we keep the same
//! externally-observable behaviour (MSB-first, big-endian RBSP order) with a
//! straightforward bit-position model. The reader consumes **RBSP** (the NAL
//! layer removes emulation-prevention bytes before constructing it).

mod golomb;
mod reader;
#[cfg(feature = "encoder")]
mod writer;

pub use reader::BitReader;
#[cfg(feature = "encoder")]
pub use writer::BitWriter;

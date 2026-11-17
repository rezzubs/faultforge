//! Compares sharing unchanged chunks with copying all of them.
//!
//! Every iteration clones a fault-free encoding, applies faults at a bit error
//! rate and decodes, like a fault injection run does. The variants:
//! - `full_copy` clones [`EncodedChunks`] and decodes all chunks.
//! - `overlay` clones [`CowEncodedChunks`] and decodes all chunks.
//! - `overlay_changes` clones [`CowEncodedChunks`] and only finds the decoded
//!   bytes that faults changed, see [`CowEncodedChunks::decoded_changes`].
//!
//! Criterion only measures time, so the number of bytes each clone doesn't
//! share is printed separately.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use memory::{
    BitBuffer, Fault,
    encoding::secded::{CowEncodedChunks, EncodedChunks},
};
use picker::Picker;
use rand::{SeedableRng, rngs::SmallRng};
use std::hint::black_box;

/// The number of data bytes in every benchmark.
const DATA_BYTE_COUNT: usize = 1 << 20;

/// Data bytes per chunk: 64 and 256 bit chunks.
const DATA_BYTES_PER_CHUNK: [usize; 2] = [8, 32];

const BIT_ERROR_RATES: [f64; 7] = [1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0];

/// Flip `bit_error_rate * bit_count` distinct random bits.
///
/// The bits are picked with [`Picker`] like in a fault injection run, so a bit
/// error rate of 1 flips every bit.
fn faults(bit_count: usize, bit_error_rate: f64) -> Vec<(Fault, usize)> {
    // The bit counts here are small enough to convert exactly.
    let fault_count = (bit_error_rate * bit_count as f64).round() as usize;
    let faults = Picker::new(bit_count, SmallRng::seed_from_u64(0))
        .take(fault_count)
        .map(|bit_index| (Fault::Flip, bit_index))
        .collect::<Vec<_>>();
    assert_eq!(
        faults.len(),
        fault_count,
        "the picker returns every bit once"
    );
    faults
}

fn bench_cow(c: &mut Criterion) {
    let data = (0..DATA_BYTE_COUNT)
        .map(|byte_index| u8::try_from(byte_index % 251).expect("the remainder is below 251"))
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("cow");
    // A single iteration takes up to a few hundred milliseconds at high bit
    // error rates.
    group.sample_size(10);

    for data_bytes_per_chunk in DATA_BYTES_PER_CHUNK {
        let pristine = EncodedChunks::encode(&data, data_bytes_per_chunk)
            .expect("the data is non-empty and the chunk size non-zero");
        let shared = CowEncodedChunks::encode(&data, data_bytes_per_chunk)
            .expect("the data is non-empty and the chunk size non-zero");
        let full_copy_memory_size = pristine.chunk_count() * pristine.encoded_bytes_per_chunk();
        let mut decoded = vec![0u8; DATA_BYTE_COUNT];

        for bit_error_rate in BIT_ERROR_RATES {
            let faults = faults(pristine.bit_count(), bit_error_rate);
            let parameter = |variant: &str| {
                format!(
                    "{}bit/{variant}/ber={bit_error_rate:e}",
                    data_bytes_per_chunk * 8
                )
            };

            let mut faulty = shared.clone();
            faulty.apply_faults(faults.iter().copied());
            println!(
                "cow/{}: {} bytes of overlay memory ({:.1}% of a full copy), {} chunks in the overlay",
                parameter("overlay"),
                faulty.overlay_memory_size(),
                100.0 * faulty.overlay_memory_size() as f64 / full_copy_memory_size as f64,
                faulty.overlay_chunk_count(),
            );

            group.bench_function(BenchmarkId::from_parameter(parameter("overlay")), |b| {
                b.iter(|| {
                    let mut faulty = black_box(&shared).clone();
                    faulty.apply_faults(faults.iter().copied());
                    faulty.decode_into(black_box(&mut decoded))
                })
            });
            group.bench_function(
                BenchmarkId::from_parameter(parameter("overlay_changes")),
                |b| {
                    b.iter(|| {
                        let mut faulty = black_box(&shared).clone();
                        faulty.apply_faults(faults.iter().copied());
                        faulty.decoded_changes(0..DATA_BYTE_COUNT)
                    })
                },
            );
            group.bench_function(BenchmarkId::from_parameter(parameter("full_copy")), |b| {
                b.iter(|| {
                    let mut faulty = black_box(&pristine).clone();
                    faulty.apply_faults(faults.iter().copied());
                    faulty.decode_into(black_box(&mut decoded))
                })
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_cow);
criterion_main!(benches);

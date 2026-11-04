use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use memory::encoding::secded::EncodedChunks;
use std::hint::black_box;

/// The number of data bytes in every benchmark.
const DATA_BYTE_COUNT: usize = 1 << 20;

/// Data bytes per chunk: 64 and 256 bit chunks.
const DATA_BYTES_PER_CHUNK: [usize; 2] = [8, 32];

fn bench_secded(c: &mut Criterion) {
    // The bit-by-bit encoder's speed doesn't depend on the values, but a
    // pattern keeps the parity bits from being all 0.
    let data = (0..DATA_BYTE_COUNT)
        .map(|byte_index| u8::try_from(byte_index % 251).expect("the remainder is below 251"))
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("secded");
    group.throughput(Throughput::Bytes(
        DATA_BYTE_COUNT.try_into().expect("usize fits into u64"),
    ));

    for data_bytes_per_chunk in DATA_BYTES_PER_CHUNK {
        let parameter = format!("{}bit", data_bytes_per_chunk * 8);

        group.bench_with_input(
            BenchmarkId::new("encode", &parameter),
            &data_bytes_per_chunk,
            |b, &data_bytes_per_chunk| {
                b.iter(|| EncodedChunks::encode(black_box(&data), data_bytes_per_chunk))
            },
        );

        let encoded = EncodedChunks::encode(&data, data_bytes_per_chunk)
            .expect("the data is non-empty and the chunk size non-zero");
        let mut decoded = vec![0u8; DATA_BYTE_COUNT];

        group.bench_function(BenchmarkId::new("decode", &parameter), |b| {
            b.iter(|| black_box(&encoded).decode_into(black_box(&mut decoded)))
        });

        group.bench_function(
            BenchmarkId::new("decode_with_double_errors", &parameter),
            |b| {
                b.iter(|| {
                    black_box(&encoded).decode_into_with_double_errors(black_box(&mut decoded))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_secded);
criterion_main!(benches);

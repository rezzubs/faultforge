use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use memory::{BitBuffer, ByteBuffer};
use std::{hint::black_box, time::Duration};

fn byte_chunks(input: Vec<u32>, output: &mut Vec<u32>, bytes_per_chunk: usize) {
    let chunked = input.to_byte_chunks(bytes_per_chunk).unwrap();

    let _ = chunked.copy_into_chunked(output);
}

fn dyn_chunks(input: Vec<u32>, output: &mut Vec<u32>, bytes_per_chunk: usize) {
    let chunked = input.to_dyn_chunks(bytes_per_chunk).unwrap();

    let _ = chunked.copy_into(output);
}

fn bench_fibs(c: &mut Criterion) {
    let mut group = c.benchmark_group("BitBuffer Chunking");

    for bytes_per_chunk in [2, 128] {
        for numel in [1 << 10, 1 << 20] {
            let input_buffer = vec![0u32; numel];

            let input = (bytes_per_chunk, numel);
            let input_name = format!("{}ch-{}el", bytes_per_chunk, numel);

            group.measurement_time(Duration::from_secs(10));

            group.bench_with_input(
                BenchmarkId::new("dyn", &input_name),
                &input,
                |b, (bytes_per_chunk, _numel)| {
                    let input_buffer = input_buffer.clone();
                    b.iter(|| {
                        let mut output_buffer = input_buffer.clone();
                        dyn_chunks(
                            black_box(input_buffer.clone()),
                            black_box(&mut output_buffer),
                            black_box(*bytes_per_chunk),
                        );
                        black_box(output_buffer);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("byte", &input_name),
                &input,
                |b, (bytes_per_chunk, _numel)| {
                    let input_buffer = input_buffer.clone();
                    b.iter(|| {
                        let mut output_buffer = input_buffer.clone();
                        byte_chunks(
                            black_box(input_buffer.clone()),
                            black_box(&mut output_buffer),
                            black_box(*bytes_per_chunk),
                        );
                        black_box(output_buffer);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_fibs);
criterion_main!(benches);

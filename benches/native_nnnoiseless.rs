#![cfg(not(target_family = "wasm"))]

mod common;

use common::{realistic_frames, synthetic_frame};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nnnoiseless::{DenoiseState, SubMatrix};

pub fn bench_single_frame_synthetic(c: &mut Criterion) {
    let frame = synthetic_frame();
    c.bench_function("bench_single_frame_synthetic", |b| {
        b.iter(|| {
            let mut state = DenoiseState::new();
            let mut output = [0.0f32; DenoiseState::FRAME_SIZE];
            state.process_frame(black_box(&mut output), black_box(&frame));
            black_box(output);
        })
    });
}

pub fn bench_full_stream_synthetic(c: &mut Criterion) {
    let samples: Vec<f32> = (0..48_000)
        .map(|x| {
            (x as f32 * 440.0 * 2.0 * std::f32::consts::PI / 48_000.0).sin() * i16::MAX as f32
        })
        .collect();
    let frames: Vec<Vec<f32>> = samples
        .chunks_exact(DenoiseState::FRAME_SIZE)
        .map(|chunk| chunk.to_vec())
        .collect();

    c.bench_function("bench_full_stream_synthetic", |b| {
        b.iter(|| {
            let mut state = DenoiseState::new();
            let mut output = [0.0f32; DenoiseState::FRAME_SIZE];
            for frame in &frames {
                state.process_frame(black_box(&mut output), black_box(frame));
            }
            black_box(output);
        })
    });
}

pub fn bench_single_frame_realistic(c: &mut Criterion) {
    let frames = realistic_frames();
    let frame = frames
        .first()
        .expect("realistic input must contain at least one full frame")
        .clone();

    c.bench_function("bench_single_frame_realistic", |b| {
        b.iter(|| {
            let mut state = DenoiseState::new();
            let mut output = [0.0f32; DenoiseState::FRAME_SIZE];
            state.process_frame(black_box(&mut output), black_box(&frame));
            black_box(output);
        })
    });
}

pub fn bench_full_stream_realistic(c: &mut Criterion) {
    let frames = realistic_frames();

    c.bench_function("bench_full_stream_realistic", |b| {
        b.iter(|| {
            let mut state = DenoiseState::new();
            let mut output = [0.0f32; DenoiseState::FRAME_SIZE];
            for frame in &frames {
                state.process_frame(black_box(&mut output), black_box(frame));
            }
            black_box(output);
        })
    });
}

fn make_mul_add_data(rows: usize, stride: usize) -> Vec<i8> {
    (0..(rows * stride))
        .map(|i| (((i * 13 + 7) % 127) as i16 - 63) as i8)
        .collect()
}

pub fn bench_mul_add_scalar(c: &mut Criterion) {
    let rows = 128usize;
    let stride = 128usize;
    let offset = 0usize;
    let data = make_mul_add_data(rows, stride);
    let input: Vec<f32> = (0..rows).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let matrix = SubMatrix {
        data: &data,
        stride,
        offset,
    };

    c.bench_function("bench_mul_add_scalar", |b| {
        b.iter(|| {
            let mut output = vec![0.0f32; stride];
            matrix.mul_add_scalar(black_box(&mut output), black_box(&input));
            black_box(output);
        })
    });
}

#[cfg(target_arch = "x86_64")]
pub fn bench_mul_add_avx(c: &mut Criterion) {
    if !std::is_x86_feature_detected!("avx") {
        return;
    }

    let rows = 128usize;
    let stride = 128usize;
    let offset = 0usize;
    let data = make_mul_add_data(rows, stride);
    let input: Vec<f32> = (0..rows).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let matrix = SubMatrix {
        data: &data,
        stride,
        offset,
    };

    c.bench_function("bench_mul_add_avx", |b| {
        b.iter(|| {
            let mut output = vec![0.0f32; stride];
            unsafe {
                matrix.mul_add_avx(black_box(&mut output), black_box(&input));
            }
            black_box(output);
        })
    });
}

#[cfg(not(target_arch = "x86_64"))]
pub fn bench_mul_add_avx(_c: &mut Criterion) {}

#[cfg(target_arch = "x86_64")]
pub fn bench_mul_add_avx2(c: &mut Criterion) {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let rows = 128usize;
    let stride = 128usize;
    let offset = 0usize;
    let data = make_mul_add_data(rows, stride);
    let input: Vec<f32> = (0..rows).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let matrix = SubMatrix {
        data: &data,
        stride,
        offset,
    };

    c.bench_function("bench_mul_add_avx2", |b| {
        b.iter(|| {
            let mut output = vec![0.0f32; stride];
            unsafe {
                matrix.mul_add_avx2(black_box(&mut output), black_box(&input));
            }
            black_box(output);
        })
    });
}

#[cfg(not(target_arch = "x86_64"))]
pub fn bench_mul_add_avx2(_c: &mut Criterion) {}

pub fn bench_mul_add_dispatch(c: &mut Criterion) {
    let rows = 128usize;
    let stride = 128usize;
    let offset = 0usize;
    let data = make_mul_add_data(rows, stride);
    let input: Vec<f32> = (0..rows).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect();
    let matrix = SubMatrix {
        data: &data,
        stride,
        offset,
    };

    c.bench_function("bench_mul_add_dispatch", |b| {
        b.iter(|| {
            let mut output = vec![0.0f32; stride];
            matrix.mul_add(black_box(&mut output), black_box(&input));
            black_box(output);
        })
    });
}

criterion_group!(
    benches,
    bench_single_frame_synthetic,
    bench_full_stream_synthetic,
    bench_single_frame_realistic,
    bench_full_stream_realistic,
    bench_mul_add_scalar,
    bench_mul_add_avx,
    bench_mul_add_avx2,
    bench_mul_add_dispatch,
);
criterion_main!(benches);

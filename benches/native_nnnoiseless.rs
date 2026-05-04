#![cfg(not(target_family = "wasm"))]

mod common;

use common::{realistic_frames, synthetic_frame};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nnnoiseless::DenoiseState;

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

criterion_group!(
    benches,
    bench_single_frame_synthetic,
    bench_full_stream_synthetic,
    bench_single_frame_realistic,
    bench_full_stream_realistic,
);
criterion_main!(benches);

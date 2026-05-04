use nnnoiseless::DenoiseState;

pub fn synthetic_frame() -> Vec<f32> {
    (0..DenoiseState::FRAME_SIZE)
        .map(|x| {
            (x as f32 * 440.0 * 2.0 * std::f32::consts::PI / 48_000.0).sin() * i16::MAX as f32
        })
        .collect()
}

pub fn realistic_frames() -> Vec<Vec<f32>> {
    let bytes = include_bytes!("../test_data/testing.raw");
    let samples: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32)
        .collect();

    samples
        .chunks_exact(DenoiseState::FRAME_SIZE)
        .map(|chunk| chunk.to_vec())
        .collect()
}

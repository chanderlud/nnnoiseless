//! Batch denoise WAV files in a directory.
//!
//! Usage:
//! `cargo run --example batch_denoise_wav -- <INPUT_DIR> <OUTPUT_DIR> [--model <MODEL_PATH>]`

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use nnnoiseless::{DenoiseState, RnnModel};

const FRAME_SIZE: usize = DenoiseState::FRAME_SIZE;
const TARGET_SAMPLE_RATE: u32 = 48_000;

fn main() -> Result<()> {
    let (input_dir, output_dir, model) = parse_args()?;
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create output directory {:?}", output_dir))?;

    let mut wav_files = gather_wav_files(&input_dir)?;
    if wav_files.is_empty() {
        bail!("No .wav files found in {:?}", input_dir);
    }
    wav_files.sort();

    for input_path in wav_files {
        let output_path = output_dir.join(
            input_path
                .file_name()
                .ok_or_else(|| anyhow!("Input file has no filename: {:?}", input_path))?,
        );
        process_wav_file(&input_path, &output_path, &model)?;
        println!("Wrote {:?}", output_path);
    }

    Ok(())
}

fn parse_args() -> Result<(PathBuf, PathBuf, RnnModel)> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 && args.len() != 5 {
        bail!(
            "Usage: {} <INPUT_DIR> <OUTPUT_DIR> [--model <MODEL_PATH>]",
            args[0]
        );
    }

    let input_dir = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);
    if !input_dir.is_dir() {
        bail!("Input path is not a directory: {:?}", input_dir);
    }

    let model = if args.len() == 5 {
        if args[3] != "--model" {
            bail!(
                "Unknown argument {:?}. Expected --model <MODEL_PATH>.",
                args[3]
            );
        }
        let bytes = fs::read(&args[4]).with_context(|| format!("Failed to read {:?}", args[4]))?;
        RnnModel::from_bytes(&bytes).context("Failed to parse model file")?
    } else {
        RnnModel::default()
    };

    Ok((input_dir, output_dir, model))
}

fn gather_wav_files(input_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut wav_files = Vec::new();
    for entry in fs::read_dir(input_dir)
        .with_context(|| format!("Failed to read directory {:?}", input_dir))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
        {
            wav_files.push(path);
        }
    }
    Ok(wav_files)
}

fn process_wav_file(input_path: &Path, output_path: &Path, model: &RnnModel) -> Result<()> {
    let reader = WavReader::open(input_path)
        .with_context(|| format!("Failed to open input wav {:?}", input_path))?;
    let spec = reader.spec();
    if spec.sample_rate != TARGET_SAMPLE_RATE {
        bail!(
            "Unsupported sample rate {} in {:?}; expected {} Hz",
            spec.sample_rate,
            input_path,
            TARGET_SAMPLE_RATE
        );
    }

    let channels = spec.channels as usize;
    let per_channel = split_to_channels(reader, spec)?;
    let denoised_per_channel: Vec<Vec<f32>> = per_channel
        .iter()
        .map(|samples| denoise_channel(samples, model))
        .collect();

    let output_spec = WavSpec {
        channels: spec.channels,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(output_path, output_spec)
        .with_context(|| format!("Failed to create output wav {:?}", output_path))?;

    let frames = denoised_per_channel
        .iter()
        .map(Vec::len)
        .min()
        .unwrap_or(0);
    for frame_idx in 0..frames {
        for ch in 0..channels {
            let sample = denoised_per_channel[ch][frame_idx]
                .max(i16::MIN as f32)
                .min(i16::MAX as f32)
                .round() as i16;
            writer.write_sample(sample)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

fn split_to_channels<R: std::io::Read>(
    reader: WavReader<R>,
    spec: WavSpec,
) -> Result<Vec<Vec<f32>>> {
    let channels = spec.channels as usize;
    let mut data = vec![Vec::<f32>::new(); channels];
    match spec.sample_format {
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if bits == 0 || bits > 32 {
                bail!("Unsupported integer bit depth: {}", bits);
            }
            for (idx, sample) in reader.into_samples::<i32>().enumerate() {
                let sample = sample?;
                let converted = if bits < 16 {
                    (sample << (16 - bits)) as f32
                } else {
                    (sample >> (bits - 16)) as f32
                };
                data[idx % channels].push(converted);
            }
        }
        SampleFormat::Float => {
            for (idx, sample) in reader.into_samples::<f32>().enumerate() {
                let sample = sample?;
                data[idx % channels].push(sample * 32767.0);
            }
        }
    }
    Ok(data)
}

fn denoise_channel(samples: &[f32], model: &RnnModel) -> Vec<f32> {
    let mut state = DenoiseState::with_model(model);
    let mut output = Vec::with_capacity(samples.len().saturating_sub(FRAME_SIZE));
    let mut out_buf = [0.0; FRAME_SIZE];
    let mut first = true;

    for frame in samples.chunks_exact(FRAME_SIZE) {
        state.process_frame(&mut out_buf, frame);
        if !first {
            output.extend_from_slice(&out_buf);
        }
        first = false;
    }

    output
}

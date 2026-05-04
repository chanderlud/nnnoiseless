//! Structures for computing audio features.
//!
//! This module contains utilities for computing features of an audio signal. These features are
//! used in two ways: they can be fed into a trained neural net for noise removal and speech
//! detection, or when the `train` feature is enabled they can be collected and used to train new
//! neural nets.

use crate::{
    common, Complex, CEPS_MEM, FRAME_SIZE, FREQ_SIZE, NB_BANDS, NB_DELTA_CEPS, NB_FEATURES,
    PITCH_BUF_SIZE, WINDOW_SIZE,
};
use once_cell::sync::OnceCell;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Contains the necessary state to compute the features of audio input and synthesize the output.
///
/// This is quite a large struct and should probably be kept behind some kind of pointer.
#[derive(Clone)]
pub struct DenoiseFeatures {
    /// Mirrored ring buffer storing recent input history in two halves.
    ///
    /// The second half mirrors the first half to keep recent windows contiguous for the common
    /// no-wrap case.
    input_mem: [f32; INPUT_MEM_STORAGE_SIZE],
    input_mem_head: usize,
    /// This is some sort of ring buffer, storing the last bunch of cepstra.
    cepstral_mem: [[f32; crate::NB_BANDS]; crate::CEPS_MEM],
    /// The index pointing to the most recent cepstrum in `cepstral_mem`. The previous cepstra are
    /// at indices mem_id - 1, mem_id - 2, etc (wrapped appropriately).
    mem_id: usize,
    mem_hp_x: [f32; 2],
    synthesis_mem: [f32; FRAME_SIZE],
    window_buf: [f32; WINDOW_SIZE],

    // What follows are various buffers. The names are cryptic, but they follow a pattern.
    /// The Fourier transform of the most recent frame of input.
    pub x: Vec<Complex>,
    /// The Fourier transform of a pitch-period-shifted window of input.
    pub p: Vec<Complex>,
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    /// The band energies of `x` (the signal).
    pub ex: [f32; NB_BANDS],
    /// The band energies of `p` (the signal, lagged by one pitch period).
    pub ep: [f32; NB_BANDS],
    /// The band correlations between `x` (the signal) and `p` (the pitch-period-lagged signal).
    pub exp: [f32; NB_BANDS],
    /// The computed features.
    features: [f32; NB_FEATURES],

    pitch_finder: crate::pitch::PitchFinder,
}

const fn max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}
const INPUT_MEM_SIZE: usize = max(FRAME_SIZE, PITCH_BUF_SIZE);
const INPUT_MEM_STORAGE_SIZE: usize = INPUT_MEM_SIZE * 2;
type FftPlans = (
    Arc<dyn RealToComplex<f32>>,
    Arc<dyn ComplexToReal<f32>>,
);
static FFT_PLANS: OnceCell<FftPlans> = OnceCell::new();

fn fft_plans() -> &'static FftPlans {
    FFT_PLANS.get_or_init(|| {
        let mut planner = RealFftPlanner::<f32>::new();
        (
            planner.plan_fft_forward(WINDOW_SIZE),
            planner.plan_fft_inverse(WINDOW_SIZE),
        )
    })
}

impl DenoiseFeatures {
    /// Creates a new, empty, `DenoiseFeatures`.
    pub fn new() -> DenoiseFeatures {
        let (fft_forward, fft_inverse) = fft_plans();
        DenoiseFeatures {
            input_mem: [0.0; INPUT_MEM_STORAGE_SIZE],
            input_mem_head: 0,
            cepstral_mem: [[0.0; NB_BANDS]; CEPS_MEM],
            mem_id: 0,
            mem_hp_x: [0.0; 2],
            synthesis_mem: [0.0; FRAME_SIZE],
            window_buf: [0.0; WINDOW_SIZE],
            x: vec![Complex::default(); FREQ_SIZE],
            p: vec![Complex::default(); FREQ_SIZE],
            fft_forward: Arc::clone(fft_forward),
            fft_inverse: Arc::clone(fft_inverse),
            ex: [0.0; NB_BANDS],
            ep: [0.0; NB_BANDS],
            exp: [0.0; NB_BANDS],
            features: [0.0; NB_FEATURES],
            pitch_finder: crate::pitch::PitchFinder::new(),
        }
    }

    /// Returns the computed features.
    pub fn features(&self) -> &[f32] {
        &self.features[..]
    }

    /// Shifts our input buffer and adds the new input to it. This is mainly used when generating
    /// training data: when running the noise reduction we use [`DenoiseFeatures::shift_and_filter_input`]
    /// instead.
    pub fn shift_input(&mut self, input: &[f32]) {
        assert!(input.len() == FRAME_SIZE);
        let write_start = self.input_mem_head;
        self.input_mem_head = (self.input_mem_head + FRAME_SIZE) % INPUT_MEM_SIZE;
        let first_len = (INPUT_MEM_SIZE - write_start).min(FRAME_SIZE);
        self.input_mem[write_start..write_start + first_len].copy_from_slice(&input[..first_len]);
        self.mirror_primary_segment(write_start, first_len);
        if first_len < FRAME_SIZE {
            let second_len = FRAME_SIZE - first_len;
            self.input_mem[..second_len].copy_from_slice(&input[first_len..]);
            self.mirror_primary_segment(0, second_len);
        }
    }

    /// Shifts our input buffer and adds the new input to it, while running the input through a
    /// high-pass filter.
    pub fn shift_and_filter_input(&mut self, input: &[f32]) {
        assert!(input.len() == FRAME_SIZE);
        let write_start = self.input_mem_head;
        self.input_mem_head = (self.input_mem_head + FRAME_SIZE) % INPUT_MEM_SIZE;
        let first_len = (INPUT_MEM_SIZE - write_start).min(FRAME_SIZE);
        crate::util::BIQUAD_HP.filter(
            &mut self.input_mem[write_start..write_start + first_len],
            &mut self.mem_hp_x,
            &input[..first_len],
        );
        self.mirror_primary_segment(write_start, first_len);
        if first_len < FRAME_SIZE {
            let second_len = FRAME_SIZE - first_len;
            crate::util::BIQUAD_HP.filter(
                &mut self.input_mem[..second_len],
                &mut self.mem_hp_x,
                &input[first_len..],
            );
            self.mirror_primary_segment(0, second_len);
        }
    }

    fn find_pitch(&mut self) -> usize {
        let input = recent_input_slice(&self.input_mem, self.input_mem_head, PITCH_BUF_SIZE, 0);
        let (pitch, _gain) = self.pitch_finder.process(input);
        pitch
    }

    /// Computes the features of the current frame.
    ///
    /// The return value is `true` if the input was pretty much silent.
    pub fn compute_frame_features(&mut self) -> bool {
        let mut ly = [0.0; NB_BANDS];
        let mut tmp = [0.0; NB_BANDS];

        transform_input(
            recent_input_slice(&self.input_mem, self.input_mem_head, WINDOW_SIZE, 0),
            &mut self.window_buf,
            &self.fft_forward,
            &mut self.x,
            &mut self.ex,
        );
        let pitch_idx = self.find_pitch();

        transform_input(
            recent_input_slice(&self.input_mem, self.input_mem_head, WINDOW_SIZE, pitch_idx),
            &mut self.window_buf,
            &self.fft_forward,
            &mut self.p,
            &mut self.ep,
        );
        crate::compute_band_corr(&mut self.exp[..], &self.x[..], &self.p[..]);
        for i in 0..NB_BANDS {
            self.exp[i] /= (0.001 + self.ex[i] * self.ep[i]).sqrt();
        }
        crate::dct(&mut tmp[..], &self.exp[..]);
        for i in 0..NB_DELTA_CEPS {
            self.features[NB_BANDS + 2 * NB_DELTA_CEPS + i] = tmp[i];
        }

        self.features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3;
        self.features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9;
        self.features[NB_BANDS + 3 * NB_DELTA_CEPS] = 0.01 * (pitch_idx as f32 - 300.0);
        let mut log_max = -2.0;
        let mut follow = -2.0;
        let mut e = 0.0;
        for i in 0..NB_BANDS {
            ly[i] = (1e-2 + self.ex[i])
                .log10()
                .max(log_max - 7.0)
                .max(follow - 1.5);
            log_max = log_max.max(ly[i]);
            follow = (follow - 1.5).max(ly[i]);
            e += self.ex[i];
        }

        if e < 0.04 {
            /* If there's no audio, avoid messing up the state. */
            for i in 0..NB_FEATURES {
                self.features[i] = 0.0;
            }
            return true;
        }
        crate::dct(&mut self.features, &ly[..]);
        self.features[0] -= 12.0;
        self.features[1] -= 4.0;
        let ceps_0_idx = self.mem_id;
        let ceps_1_idx = if self.mem_id < 1 {
            CEPS_MEM + self.mem_id - 1
        } else {
            self.mem_id - 1
        };
        let ceps_2_idx = if self.mem_id < 2 {
            CEPS_MEM + self.mem_id - 2
        } else {
            self.mem_id - 2
        };

        for i in 0..NB_BANDS {
            self.cepstral_mem[ceps_0_idx][i] = self.features[i];
        }
        self.mem_id += 1;

        let ceps_0 = &self.cepstral_mem[ceps_0_idx];
        let ceps_1 = &self.cepstral_mem[ceps_1_idx];
        let ceps_2 = &self.cepstral_mem[ceps_2_idx];
        for i in 0..NB_DELTA_CEPS {
            self.features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
            self.features[NB_BANDS + i] = ceps_0[i] - ceps_2[i];
            self.features[NB_BANDS + NB_DELTA_CEPS + i] = ceps_0[i] - 2.0 * ceps_1[i] + ceps_2[i];
        }

        /* Spectral variability features. */
        let mut spec_variability = 0.0;
        if self.mem_id == CEPS_MEM {
            self.mem_id = 0;
        }
        for i in 0..CEPS_MEM {
            let mut min_dist = 1e15f32;
            for j in 0..CEPS_MEM {
                let mut dist = 0.0;
                for k in 0..NB_BANDS {
                    let tmp = self.cepstral_mem[i][k] - self.cepstral_mem[j][k];
                    dist += tmp * tmp;
                }
                if j != i {
                    min_dist = min_dist.min(dist);
                }
            }
            spec_variability += min_dist;
        }

        self.features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM as f32 - 2.1;

        false
    }

    /// Applies a filter to the audio, attenuating pitches that have poor correlation with the
    /// pitch-lagged signal.
    pub fn pitch_filter(&mut self, gain: &[f32; NB_BANDS]) {
        let mut r = [0.0; NB_BANDS];
        let mut rf = [0.0; FREQ_SIZE];
        for i in 0..NB_BANDS {
            r[i] = if self.exp[i] > gain[i] {
                1.0
            } else {
                let exp_sq = self.exp[i] * self.exp[i];
                let g_sq = gain[i] * gain[i];
                exp_sq * (1.0 - g_sq) / (0.001 + g_sq * (1.0 - exp_sq))
            };
            r[i] = r[i].clamp(0.0, 1.0).sqrt();
            r[i] *= (self.ex[i] / (1e-8 + self.ep[i])).sqrt();
        }
        crate::interp_band_gain(&mut rf[..], &r[..]);
        let rf: &mut [f32] = &mut rf;
        for i in 0..FREQ_SIZE {
            self.x[i] += self.p[i] * rf[i];
        }

        let mut new_e = [0.0; NB_BANDS];
        crate::compute_band_corr(&mut new_e[..], &self.x, &self.x);
        for i in 0..NB_BANDS {
            r[i] = (self.ex[i] / (1e-8 + new_e[i])).sqrt();
        }
        crate::interp_band_gain(&mut rf[..], &r[..]);
        for i in 0..FREQ_SIZE {
            self.x[i] *= rf[i];
        }
    }

    pub(crate) fn apply_gain(&mut self, gain: &[f32; FREQ_SIZE]) {
        for (x, g) in self.x.iter_mut().zip(gain.iter()) {
            *x *= *g;
        }
    }

    pub(crate) fn frame_synthesis(&mut self, out: &mut [f32]) {
        let fft_inverse = Arc::clone(&self.fft_inverse);
        fft_inverse
            .process(&mut self.x, &mut self.window_buf)
            .expect("inverse real FFT failed");
        // Not too sure why this scaling factor is introduced
        for x in &mut self.window_buf {
            *x /= 2.0;
        }

        crate::apply_window_in_place(&mut self.window_buf[..]);
        for i in 0..FRAME_SIZE {
            out[i] = self.window_buf[i] + self.synthesis_mem[i];
            self.synthesis_mem[i] = self.window_buf[FRAME_SIZE + i];
        }
    }

    fn mirror_primary_segment(&mut self, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        let (primary, mirrored) = self.input_mem.split_at_mut(INPUT_MEM_SIZE);
        mirrored[start..start + len].copy_from_slice(&primary[start..start + len]);
    }

}

/// Fourier transforms the input.
///
/// The Fourier transform goes in `x` and the band energies go in `ex`.
fn transform_input(
    input: &[f32],
    window_buf: &mut [f32; WINDOW_SIZE],
    fft_forward: &Arc<dyn RealToComplex<f32>>,
    x: &mut [Complex],
    ex: &mut [f32],
) {
    debug_assert_eq!(input.len(), WINDOW_SIZE);
    debug_assert_eq!(x.len(), FREQ_SIZE);
    window_buf.copy_from_slice(input);
    crate::apply_window_in_place(window_buf);
    fft_forward
        .process(window_buf, x)
        .expect("forward real FFT failed");

    // In the original RNNoise code, the forward transform is normalized and the inverse
    // transform isn't. `realfft` doesn't normalize either one, so we do it ourselves.
    let norm = common().wnorm;
    for i in 0..FREQ_SIZE {
        x[i] *= norm;
    }

    crate::compute_band_corr(ex, x, x);
}

fn recent_input_slice<'a>(
    input_mem: &'a [f32; INPUT_MEM_STORAGE_SIZE],
    input_head: usize,
    len: usize,
    lag: usize,
) -> &'a [f32] {
    assert!(len + lag <= INPUT_MEM_SIZE);
    let start = INPUT_MEM_SIZE - (len + lag);
    let idx = input_head + start;
    &input_mem[idx..idx + len]
}

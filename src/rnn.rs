use std::borrow::Cow;

use crate::util::{relu, sigmoid_approx, tansig_approx, zip3};

const MAX_NEURONS: usize = 128;

// It's annoying to expose a public API with `i8`s, because `include_bytes` works with `u8`s only.
// So we do conversions from `&[i8]` to `&[u8]` internally. Hopefully at some point rust will have
// a safe API for this...
fn to_i8(x: &[u8]) -> &[i8] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const i8, x.len()) }
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Tanh = 0,
    Sigmoid = 1,
    Relu = 2,
}

const WEIGHTS_SCALE: f32 = 1.0 / 256.0;

#[derive(Clone)]
pub struct DenseLayer {
    /// An array of length `nb_neurons`.
    pub bias: Cow<'static, [i8]>,
    /// An array of length `nb_inputs * nb_neurons`.
    pub input_weights: Cow<'static, [i8]>,
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    pub activation: Activation,
}

#[derive(Clone)]
pub struct GruLayer {
    /// An array of length `3 * nb_neurons`.
    pub bias: Cow<'static, [i8]>,
    /// An array of length `3 * nb_inputs * nb_neurons`.
    pub input_weights: Cow<'static, [i8]>,
    /// An array of length `3 * nb_neurons^2`.
    pub recurrent_weights: Cow<'static, [i8]>,
    pub nb_inputs: usize,
    pub nb_neurons: usize,
    pub activation: Activation,
}

/// An `RnnModel` contains all the model parameters for the denoising algorithm.
/// `nnnoiseless` has a built-in model that should work for most purposes, but if you have
/// specific needs then you might benefit from training a custom model. Scripts for model
/// training are available as part of [`RNNoise`]; once the model is trained, you can load it
/// here.
///
/// [`RNNoise`]: https://github.com/xiph/rnnoise
#[derive(Clone)]
pub struct RnnModel {
    pub(crate) input_dense: DenseLayer,
    pub(crate) vad_gru: GruLayer,
    pub(crate) noise_gru: GruLayer,
    pub(crate) denoise_gru: GruLayer,
    pub(crate) denoise_output: DenseLayer,
    pub(crate) vad_output: DenseLayer,
}

#[derive(Clone)]
pub struct RnnState<'model> {
    model: Cow<'model, RnnModel>,
    vad_gru_state: Vec<f32>,
    noise_gru_state: Vec<f32>,
    denoise_gru_state: Vec<f32>,
}

impl RnnModel {
    /// Reads an `RnnModel` from an array of bytes, in the format produced by the
    /// `nnnoiseless` training scripts.
    pub fn from_bytes(bytes: &[u8]) -> Option<RnnModel> {
        RnnModel::from_bytes_impl(to_i8(bytes), |xs| Cow::Owned(xs.to_owned()))
    }

    /// Reads an `RnnModel` from a static array of bytes, in the format produced by the
    /// `nnnoiseless` training scripts.
    ///
    /// This differs from [`RnnModel::from_bytes`] in that the returned model doesn't need to
    /// allocate its own byte buffers; it will just store references to the provided `bytes` array.
    ///
    /// For example, if you have your neural network weights available at compile-time then the
    /// following code will embed them into your binary and initialize a model without allocation:
    ///
    /// ```ignore
    /// let weight_data: &'static [u8] = include_bytes!("/path/to/model/weights.rnn");
    /// let model = RnnModel::from_static_bytes(weight_data).expect("Corrupted model file");
    /// ```
    pub fn from_static_bytes(bytes: &'static [u8]) -> Option<RnnModel> {
        RnnModel::from_bytes_impl(to_i8(bytes), Cow::Borrowed)
    }

    /// Reads an `RnnModel` from an array of bytes, in our new nnnoiseless format.
    ///
    /// The format is simple: each NN layer is represented by an array of signed `i8`'s,
    /// and these layers as simply concatenated.
    ///
    /// The format for a dense layer is
    /// <nb_neurons> <nb_inputs> <activation>
    /// <weights...>
    /// <bias...>
    /// where each of the <?> terms represents a single integer, and each of the <?...> terms
    /// represents an array of integers of the appropriate length (`weights` has length
    /// `nb_neurons * nb_inputs` and `bias` has length `nb_neurons`).
    ///
    /// The format for a GRU layer is
    /// <nb_neurons> <nb_inputs> <activation>
    /// <input_weights...>
    /// <recurrent_weights...>
    /// <bias...>
    /// where `input_weights` and `recurrent_weights` have length `3 * nb_inputs * nb_neurons` each,
    /// and `bias` has length `3 * nb_neurons`.
    fn from_bytes_impl<'a>(
        bytes: &'a [i8],
        moo: fn(&'a [i8]) -> Cow<'static, [i8]>,
    ) -> Option<RnnModel> {
        let read_array = |bytes: &'a [i8], len: usize| -> Option<(Cow<'static, [i8]>, &[i8])> {
            if bytes.len() >= len {
                Some((moo(&bytes[..len]), &bytes[len..]))
            } else {
                None
            }
        };

        fn unsigned(b: i8) -> Option<usize> {
            if b >= 0 {
                Some(b as usize)
            } else {
                None
            }
        }

        fn act(x: i8) -> Option<Activation> {
            match x {
                0 => Some(Activation::Tanh),
                1 => Some(Activation::Sigmoid),
                2 => Some(Activation::Relu),
                _ => None,
            }
        }

        let read_dense = |bytes: &'a [i8]| -> Option<(DenseLayer, &[i8])> {
            if bytes.len() < 3 {
                return None;
            }

            let nb_inputs = unsigned(bytes[0])?;
            let nb_neurons = unsigned(bytes[1])?;
            let activation = act(bytes[2])?;
            let (input_weights, bytes) = read_array(&bytes[3..], nb_neurons * nb_inputs)?;
            let (bias, bytes) = read_array(bytes, nb_neurons)?;

            let layer = DenseLayer {
                nb_inputs,
                nb_neurons,
                input_weights,
                bias,
                activation,
            };
            Some((layer, bytes))
        };

        let read_gru = |bytes: &'a [i8]| -> Option<(GruLayer, &[i8])> {
            if bytes.len() < 3 {
                return None;
            }

            let nb_inputs = unsigned(bytes[0])?;
            let nb_neurons = unsigned(bytes[1])?;
            let activation = act(bytes[2])?;
            let (input_weights, bytes) = read_array(&bytes[3..], 3 * nb_neurons * nb_inputs)?;
            let (recurrent_weights, bytes) = read_array(bytes, 3 * nb_neurons * nb_neurons)?;
            let (bias, bytes) = read_array(bytes, 3 * nb_neurons)?;

            let layer = GruLayer {
                nb_inputs,
                nb_neurons,
                input_weights,
                recurrent_weights,
                bias,
                activation,
            };
            Some((layer, bytes))
        };

        let (input_dense, bytes) = read_dense(bytes)?;
        let (vad_gru, bytes) = read_gru(bytes)?;
        let (noise_gru, bytes) = read_gru(bytes)?;
        let (denoise_gru, bytes) = read_gru(bytes)?;
        let (denoise_output, bytes) = read_dense(bytes)?;
        let (vad_output, bytes) = read_dense(bytes)?;

        if !bytes.is_empty() {
            return None;
        }

        // The input to the first layer must be of size 42, because that's how many features
        // there are. The denoise output must be of size 22, and the vad output must be of size 1.
        // Other than that, the output of one layer must match with the inputs of the following
        // layer.
        if input_dense.nb_inputs != 42
            || denoise_output.nb_neurons != 22
            || vad_output.nb_neurons != 1
        {
            return None;
        }
        if input_dense.nb_neurons != vad_gru.nb_inputs || vad_gru.nb_neurons != vad_output.nb_inputs
        {
            return None;
        }
        if 42 + input_dense.nb_neurons + vad_gru.nb_neurons != noise_gru.nb_inputs {
            return None;
        }
        if 42 + vad_gru.nb_neurons + noise_gru.nb_neurons != denoise_gru.nb_inputs {
            return None;
        }
        if denoise_gru.nb_neurons != denoise_output.nb_inputs {
            return None;
        }

        Some(RnnModel {
            input_dense,
            vad_gru,
            noise_gru,
            denoise_gru,
            denoise_output,
            vad_output,
        })
    }
}

impl Default for RnnModel {
    fn default() -> RnnModel {
        let bytes: &'static [u8] = include_bytes!("weights.rnn");
        RnnModel::from_static_bytes(bytes).unwrap()
    }
}

impl DenseLayer {
    fn matrix(&self) -> SubMatrix<'_> {
        SubMatrix {
            data: self.input_weights.as_ref(),
            stride: self.nb_neurons,
            offset: 0,
        }
    }

    fn compute(&self, output: &mut [f32], input: &[f32]) {
        copy_i8(output, &self.bias[..]);
        self.matrix().mul_add(output, input);

        match self.activation {
            Activation::Sigmoid => {
                for out in output.iter_mut() {
                    *out = sigmoid_approx(*out * WEIGHTS_SCALE);
                }
            }
            Activation::Tanh => {
                for out in output.iter_mut() {
                    *out = tansig_approx(*out * WEIGHTS_SCALE);
                }
            }
            Activation::Relu => {
                for out in output.iter_mut() {
                    *out = relu(*out * WEIGHTS_SCALE);
                }
            }
        }
    }
}

impl GruLayer {
    fn input_submatrix(&self, offset: usize) -> SubMatrix<'_> {
        SubMatrix {
            data: self.input_weights.as_ref(),
            stride: self.nb_neurons * 3,
            offset,
        }
    }

    fn rec_submatrix(&self, offset: usize) -> SubMatrix<'_> {
        SubMatrix {
            data: self.recurrent_weights.as_ref(),
            stride: self.nb_neurons * 3,
            offset,
        }
    }

    fn compute(&self, state: &mut [f32], input: &[f32]) {
        let mut z = [0.0; MAX_NEURONS];
        let mut r = [0.0; MAX_NEURONS];
        let mut h = [0.0; MAX_NEURONS];
        let n = self.nb_neurons;

        // Compute update gate.
        copy_i8(&mut z[0..n], &self.bias[0..n]);
        self.input_submatrix(0).mul_add(&mut z[0..n], input);
        self.rec_submatrix(0).mul_add(&mut z[0..n], &state[..]);
        for z in z[0..n].iter_mut() {
            *z = sigmoid_approx(WEIGHTS_SCALE * *z);
        }

        // Compute reset gate.
        copy_i8(&mut r[0..n], &self.bias[n..(2 * n)]);
        self.input_submatrix(n).mul_add(&mut r[0..n], input);
        self.rec_submatrix(n).mul_add(&mut r[0..n], &state[..]);
        for (out, &s) in r[0..n].iter_mut().zip(&state[..]) {
            *out = s * sigmoid_approx(WEIGHTS_SCALE * *out);
        }

        // Compute output.
        copy_i8(&mut h[0..n], &self.bias[(2 * n)..]);
        self.input_submatrix(2 * n).mul_add(&mut h[0..n], input);
        self.rec_submatrix(2 * n).mul_add(&mut h[0..n], &r[0..n]);

        for (s, &z, &h) in zip3(state, &z[0..n], &h[0..n]) {
            let h = match self.activation {
                Activation::Sigmoid => sigmoid_approx(WEIGHTS_SCALE * h),
                Activation::Tanh => tansig_approx(WEIGHTS_SCALE * h),
                Activation::Relu => relu(WEIGHTS_SCALE * h),
            };
            *s = z * *s + (1.0 - z) * h;
        }
    }
}

impl<'model> RnnState<'model> {
    pub(crate) fn new(model: Cow<'model, RnnModel>) -> RnnState<'model> {
        let vad_gru_state = vec![0.0f32; model.vad_gru.nb_neurons];
        let noise_gru_state = vec![0.0f32; model.noise_gru.nb_neurons];
        let denoise_gru_state = vec![0.0f32; model.denoise_gru.nb_neurons];
        RnnState {
            model,
            vad_gru_state,
            noise_gru_state,
            denoise_gru_state,
        }
    }

    pub fn compute(&mut self, gains: &mut [f32], vad: &mut [f32], input: &[f32]) {
        assert_eq!(input.len(), INPUT_SIZE);

        let mut buf = [0.0; MAX_NEURONS * 3];
        let mut denoise_buf = [0.0; MAX_NEURONS * 3];
        let model = &self.model;

        let vad_gru_state = &mut self.vad_gru_state[..];
        let noise_gru_state = &mut self.noise_gru_state[..];
        let denoise_gru_state = &mut self.denoise_gru_state[..];
        model
            .input_dense
            .compute(&mut buf[0..model.input_dense.nb_neurons], input);
        model
            .vad_gru
            .compute(vad_gru_state, &buf[0..model.input_dense.nb_neurons]);
        model.vad_output.compute(vad, vad_gru_state);

        copy(&mut buf[model.input_dense.nb_neurons..], vad_gru_state);
        copy(
            &mut buf[(model.input_dense.nb_neurons + model.vad_gru.nb_neurons)..],
            input,
        );
        model.noise_gru.compute(noise_gru_state, &buf);

        copy(&mut denoise_buf, vad_gru_state);
        copy(
            &mut denoise_buf[model.vad_gru.nb_neurons..],
            noise_gru_state,
        );
        copy(
            &mut denoise_buf[(model.vad_gru.nb_neurons + model.noise_gru.nb_neurons)..],
            input,
        );
        model.denoise_gru.compute(denoise_gru_state, &denoise_buf);
        model.denoise_output.compute(gains, denoise_gru_state);
    }
}

const INPUT_SIZE: usize = 42;

fn copy(dst: &mut [f32], src: &[f32]) {
    for (x, y) in dst.iter_mut().zip(src) {
        *x = *y;
    }
}

fn copy_i8(dst: &mut [f32], src: &[i8]) {
    for (x, y) in dst.iter_mut().zip(src) {
        *x = *y as f32;
    }
}

struct SubMatrix<'a> {
    data: &'a [i8],
    stride: usize,
    offset: usize,
}

impl<'a> SubMatrix<'a> {
    fn mul_add(&self, output: &mut [f32], input: &[f32]) {
        #[cfg(target_arch = "wasm32")]
        unsafe {
            self.mul_add_wasm(output, input);
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.mul_add_scalar(output, input);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn mul_add_scalar(&self, output: &mut [f32], input: &[f32]) {
        let rows = self.data.len() / self.stride;
        let input = &input[..input.len().min(rows)];
        let base = self.offset;
        let output_len = output.len().min(self.stride.saturating_sub(base));
        for (neuron_idx, out) in output.iter_mut().take(output_len).enumerate() {
            let mut acc = 0.0f32;
            let mut in_idx = 0usize;
            while in_idx + 4 <= input.len() {
                let w0 = self.data[in_idx * self.stride + base + neuron_idx] as f32;
                let w1 = self.data[(in_idx + 1) * self.stride + base + neuron_idx] as f32;
                let w2 = self.data[(in_idx + 2) * self.stride + base + neuron_idx] as f32;
                let w3 = self.data[(in_idx + 3) * self.stride + base + neuron_idx] as f32;
                acc += w0 * input[in_idx]
                    + w1 * input[in_idx + 1]
                    + w2 * input[in_idx + 2]
                    + w3 * input[in_idx + 3];
                in_idx += 4;
            }
            while in_idx < input.len() {
                acc += self.data[in_idx * self.stride + base + neuron_idx] as f32 * input[in_idx];
                in_idx += 1;
            }
            *out += acc;
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[target_feature(enable = "simd128")]
    unsafe fn mul_add_wasm(&self, output: &mut [f32], input: &[f32]) {
        use core::arch::wasm32::{
            f32x4_add, f32x4_convert_i32x4, f32x4_mul, f32x4_splat, i16x8_extend_high_i8x16,
            i16x8_extend_low_i8x16, i32x4_extend_high_i16x8, i32x4_extend_low_i16x8, v128_load,
            v128_store,
        };

        let rows = self.data.len() / self.stride;
        let input = &input[..input.len().min(rows)];
        let base = self.offset;
        let output_len = output.len().min(self.stride.saturating_sub(base));
        let mut neuron_idx = 0usize;
        let chunk = 16usize;
        let data_ptr = self.data.as_ptr();

        while neuron_idx + chunk <= output_len {
            let mut acc0 = f32x4_splat(0.0);
            let mut acc1 = f32x4_splat(0.0);
            let mut acc2 = f32x4_splat(0.0);
            let mut acc3 = f32x4_splat(0.0);

            for (in_idx, &x) in input.iter().enumerate() {
                let row_start = in_idx * self.stride + base + neuron_idx;
                let w_i8 = v128_load(data_ptr.add(row_start) as *const _);
                let w_i16_lo = i16x8_extend_low_i8x16(w_i8);
                let w_i16_hi = i16x8_extend_high_i8x16(w_i8);
                let w0 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(w_i16_lo));
                let w1 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(w_i16_lo));
                let w2 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(w_i16_hi));
                let w3 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(w_i16_hi));
                let x4 = f32x4_splat(x);
                acc0 = f32x4_add(acc0, f32x4_mul(w0, x4));
                acc1 = f32x4_add(acc1, f32x4_mul(w1, x4));
                acc2 = f32x4_add(acc2, f32x4_mul(w2, x4));
                acc3 = f32x4_add(acc3, f32x4_mul(w3, x4));
            }

            let out_ptr = output.as_mut_ptr().add(neuron_idx);
            let out0 = v128_load(out_ptr as *const _);
            let out1 = v128_load(out_ptr.add(4) as *const _);
            let out2 = v128_load(out_ptr.add(8) as *const _);
            let out3 = v128_load(out_ptr.add(12) as *const _);
            v128_store(out_ptr as *mut _, f32x4_add(out0, acc0));
            v128_store(out_ptr.add(4) as *mut _, f32x4_add(out1, acc1));
            v128_store(out_ptr.add(8) as *mut _, f32x4_add(out2, acc2));
            v128_store(out_ptr.add(12) as *mut _, f32x4_add(out3, acc3));

            neuron_idx += chunk;
        }

        if neuron_idx < output_len {
            for (local_idx, out) in output[neuron_idx..output_len].iter_mut().enumerate() {
                let mut acc = 0.0f32;
                let idx = base + neuron_idx + local_idx;
                for (in_idx, &x) in input.iter().enumerate() {
                    acc += self.data[in_idx * self.stride + idx] as f32 * x;
                }
                *out += acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SubMatrix;

    fn reference_mul_add(data: &[i8], stride: usize, offset: usize, output: &mut [f32], input: &[f32]) {
        let rows = data.len() / stride;
        let input_len = input.len().min(rows);
        let output_len = output.len().min(stride.saturating_sub(offset));
        for neuron_idx in 0..output_len {
            let mut acc = 0.0f32;
            for in_idx in 0..input_len {
                acc += data[in_idx * stride + offset + neuron_idx] as f32 * input[in_idx];
            }
            output[neuron_idx] += acc;
        }
    }

    fn assert_equal(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual, expected);
    }

    fn make_data(rows: usize, stride: usize) -> Vec<i8> {
        (0..(rows * stride))
            .map(|i| (((i * 13 + 7) % 127) as i16 - 63) as i8)
            .collect()
    }

    fn run_mul_add_matches_reference_case() {
        let rows = 5;
        let stride = 21;
        let offset = 2;
        let data = make_data(rows, stride);
        let input = vec![0.25, -1.0, 3.0, -0.75, 2.5, 11.0];
        let mut actual = vec![1.0f32; 19];
        let mut expected = actual.clone();
        let matrix = SubMatrix {
            data: &data,
            stride,
            offset,
        };

        reference_mul_add(&data, stride, offset, &mut expected, &input);
        matrix.mul_add(&mut actual, &input);
        assert_equal(&actual, &expected);
    }

    fn run_mul_add_respects_output_bounds_case() {
        let rows = 4;
        let stride = 10;
        let offset = 9;
        let data = make_data(rows, stride);
        let input = vec![1.5, -2.0, 0.5, 3.0];
        let mut actual = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut expected = actual.clone();
        let matrix = SubMatrix {
            data: &data,
            stride,
            offset,
        };

        reference_mul_add(&data, stride, offset, &mut expected, &input);
        matrix.mul_add(&mut actual, &input);
        assert_equal(&actual, &expected);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn mul_add_matches_reference_native() {
        run_mul_add_matches_reference_case();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn mul_add_respects_output_bounds_native() {
        run_mul_add_respects_output_bounds_case();
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn mul_add_matches_reference_wasm() {
        run_mul_add_matches_reference_case();
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test::wasm_bindgen_test]
    fn mul_add_respects_output_bounds_wasm() {
        run_mul_add_respects_output_bounds_case();
    }
}

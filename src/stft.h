#pragma once
/**
 * stft.h - STFT/ISTFT implementation
 * 
 * Implements:
 * - Hann window generation
 * - Center padding (reflect mode)
 * - Frame extraction
 * - Radix-2 Cooley-Tukey FFT
 * - Real-to-complex FFT (rfft)
 * - Inverse FFT (irfft)
 * - Full STFT/ISTFT matching torch.stft/torch.istft
 */

#include <cmath>
#include <vector>
#include <complex>
#include <cstring>
#include <algorithm> // for std::swap

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace stft {

// Complex number type
using Complex = std::complex<float>;

//=============================================================================
// Window Functions
//=============================================================================

/**
 * Generate Hann window matching torch.hann_window()
 * PyTorch uses periodic=True by default for STFT compatibility
 * Periodic formula: 0.5 * (1 - cos(2*pi*n / N))
 * Symmetric formula: 0.5 * (1 - cos(2*pi*n / (N-1)))
 */
inline void hann_window(float* out, int size, bool periodic = true) {
    int divisor = periodic ? size : (size - 1);
    for (int i = 0; i < size; ++i) {
        out[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / divisor));
    }
}

//=============================================================================
// FFT Implementation (Cooley-Tukey Radix-2)
//=============================================================================

/**
 * Bit-reversal permutation for radix-2 FFT
 */
inline void bit_reverse(Complex* data, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(data[i], data[j]);
        }
        int m = n >> 1;
        while (j >= m && m > 0) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

/**
 * In-place Cooley-Tukey radix-2 FFT
 * @param data Complex array of size n (must be power of 2)
 * @param n Size of array
 * @param inverse If true, compute inverse FFT
 */
inline void fft_radix2(Complex* data, int n, bool inverse = false) {
    bit_reverse(data, n);
    
    // Danielson-Lanczos lemma
    for (int len = 2; len <= n; len <<= 1) {
        float angle = (inverse ? 2.0f : -2.0f) * static_cast<float>(M_PI) / len;
        Complex w_n(std::cos(angle), std::sin(angle));
        
        for (int i = 0; i < n; i += len) {
            Complex w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                Complex u = data[i + j];
                Complex t = w * data[i + j + len / 2];
                data[i + j] = u + t;
                data[i + j + len / 2] = u - t;
                w *= w_n;
            }
        }
    }
    
    // Normalize for inverse FFT
    if (inverse) {
        for (int i = 0; i < n; ++i) {
            data[i] /= static_cast<float>(n);
        }
    }
}

/**
 * Real-to-complex FFT (rfft) matching torch.fft.rfft
 * @param input Real input array of size n
 * @param output Complex output array of size n/2+1
 * @param n Size of input (must be power of 2)
 */
inline void rfft(const float* input, Complex* output, int n) {
    // Copy to complex buffer
    std::vector<Complex> buffer(n);
    for (int i = 0; i < n; ++i) {
        buffer[i] = Complex(input[i], 0.0f);
    }
    
    // Compute full FFT
    fft_radix2(buffer.data(), n, false);
    
    // Extract first n/2+1 coefficients (one-sided)
    int n_out = n / 2 + 1;
    for (int i = 0; i < n_out; ++i) {
        output[i] = buffer[i];
    }
}

/**
 * Complex-to-real inverse FFT (irfft) matching torch.fft.irfft
 * @param input Complex input array of size n/2+1
 * @param output Real output array of size n
 * @param n_out Size of output (must be power of 2)
 */
inline void irfft(const Complex* input, float* output, int n_out) {
    int n_freq = n_out / 2 + 1;
    
    // Reconstruct full spectrum (conjugate symmetry)
    std::vector<Complex> buffer(n_out);
    for (int i = 0; i < n_freq; ++i) {
        buffer[i] = input[i];
    }
    for (int i = n_freq; i < n_out; ++i) {
        buffer[i] = std::conj(buffer[n_out - i]);
    }
    
    // Compute inverse FFT
    fft_radix2(buffer.data(), n_out, true);
    
    // Extract real part
    for (int i = 0; i < n_out; ++i) {
        output[i] = buffer[i].real();
    }
}

//=============================================================================
// STFT Implementation
//=============================================================================

/**
 * Short-Time Fourier Transform matching torch.stft
 * 
 * @param audio Input audio [n_samples]
 * @param n_samples Number of samples
 * @param n_fft FFT size
 * @param hop_length Hop between frames
 * @param win_length Window length
 * @param window Window function [win_length]
 * @param center If true, pad signal on both sides
 * @param output Output complex spectrogram [n_freq, n_frames, 2] (real, imag pairs)
 * @param n_frames_out Output parameter: number of frames
 */
inline void compute_stft(
    const float* audio,
    int n_samples,
    int n_fft,
    int hop_length,
    int win_length,
    const float* window,
    bool center,
    float* output,
    int* n_frames_out
) {
    // Center padding
    int pad_amount = center ? n_fft / 2 : 0;
    int padded_len = n_samples + 2 * pad_amount;
    
    std::vector<float> padded(padded_len);
    
    if (center) {
        // Reflect padding
        // Left pad (reflect)
        for (int i = 0; i < pad_amount; ++i) {
            int src_idx = pad_amount - i;
            if (src_idx >= n_samples) src_idx = n_samples - 1;
            padded[i] = audio[src_idx];
        }
        // Center (copy)
        if (n_samples > 0) {
            std::memcpy(padded.data() + pad_amount, audio, n_samples * sizeof(float));
        }
        // Right pad (reflect)
        for (int i = 0; i < pad_amount; ++i) {
            int src_idx = n_samples - 2 - i;
            if (src_idx < 0) src_idx = 0;
            padded[pad_amount + n_samples + i] = audio[src_idx];
        }
    } else {
        std::memcpy(padded.data(), audio, n_samples * sizeof(float));
    }
    
    // Calculate number of frames
    // PyTorch formula: (L - N) / H + 1
    int n_frames = 1 + (padded_len - n_fft) / hop_length;
    if (n_frames < 0) n_frames = 0;
    *n_frames_out = n_frames;
    
    // Number of output frequency bins
    int n_freq = n_fft / 2 + 1;
    
    // Prepare padded window if win_length < n_fft
    std::vector<float> window_padded(n_fft, 0.0f);
    if (win_length < n_fft) {
        int left = (n_fft - win_length) / 2;
        std::memcpy(window_padded.data() + left, window, win_length * sizeof(float));
    } else {
        std::memcpy(window_padded.data(), window, n_fft * sizeof(float));
    }
    
    // Buffers
    std::vector<float> frame(n_fft);
    std::vector<Complex> fft_out(n_freq);
    
    // Process each frame
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int f = 0; f < n_frames; ++f) {
        int start = f * hop_length;
        
        // Extract and window frame
        // Need private buffer for frame and fft_out if logical threads share memory?
        // Wait, std::vector inside loop is local to block, so essentially thread-private?
        // YES. Variables declared inside the loop are private to the iteration/thread.
        
        // However, we need to be careful about allocating vectors inside a loop in parallel (heap contention).
        // It's better to allocate buffers per thread or use raw arrays.
        // For simplicity and since n_fft is small (2048), stack array or thread_local vector is better.
        // But std::vector inside parallel for is safe but might allocate.
        // n_fft=2048 float is 8KB. 
        std::vector<float> frame(n_fft); // Allocation!
        std::vector<Complex> fft_out(n_freq);
        
        for (int i = 0; i < n_fft; ++i) {
            frame[i] = padded[start + i] * window_padded[i];
        }
        
        // Compute FFT
        rfft(frame.data(), fft_out.data(), n_fft);
        
        // Store in output [n_freq, n_frames, 2] format
        for (int k = 0; k < n_freq; ++k) {
            // Note: Output layout is [Freq, Time, 2]
            output[(k * n_frames + f) * 2 + 0] = fft_out[k].real();
            output[(k * n_frames + f) * 2 + 1] = fft_out[k].imag();
        }
    }
}

/**
 * Inverse Short-Time Fourier Transform matching torch.istft
 * 
 * @param stft_data Input complex spectrogram [n_freq, n_frames, 2]
 * @param n_freq Number of frequency bins
 * @param n_frames Number of frames
 * @param n_fft FFT size
 * @param hop_length Hop between frames
 * @param win_length Window length
 * @param window Window function [win_length]
 * @param center If true, signal was centered
 * @param length Expected output length (or 0 for auto)
 * @param output Output audio
 */
inline void compute_istft(
    const float* stft_data,
    int n_freq,
    int n_frames,
    int n_fft,
    int hop_length,
    int win_length,
    const float* window,
    bool center,
    int length,
    float* output
) {
    // Calculate expected output signal length
    int expected_len = n_fft + hop_length * (n_frames - 1);
    int pad_amount = center ? n_fft / 2 : 0;
    int output_len = (length > 0) ? length : (expected_len - 2 * pad_amount);
    
    // Prepare padded window
    std::vector<float> window_padded(n_fft, 0.0f);
    if (win_length < n_fft) {
        int left = (n_fft - win_length) / 2;
        std::memcpy(window_padded.data() + left, window, win_length * sizeof(float));
    } else {
        std::memcpy(window_padded.data(), window, n_fft * sizeof(float));
    }
    
    // Overlap-add buffers
    // This is tricky for parallelization: race condition on y (overlap-add).
    // We CANNOT parallelize the write to 'y' easily without atomic float add (slow/hard) or reduction.
    // APPROACH:
    // 1. Parallel IFFT: Compute all frames' time-domain signals into a large buffer [n_frames, n_fft].
    // 2. Serial Overlap-Add: Add them up. (Overlap-add is O(N_Frames * N_FFT), same complexity, but memory bound).
    // Serial part might be fast enough if FFT is the heavy lifter.
    // FFT is O(N log N). Overlap add is O(N). FFT dominates.
    
    // Step 1: Compute all IFFTs in parallel
    std::vector<float> frames_time_domain(n_frames * n_fft);
    
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int f = 0; f < n_frames; ++f) {
        std::vector<Complex> fft_in(n_freq);
        std::vector<float> frame_out(n_fft);
        
        // Extract complex spectrum
        for (int k = 0; k < n_freq; ++k) {
            float re = stft_data[(k * n_frames + f) * 2 + 0];
            float im = stft_data[(k * n_frames + f) * 2 + 1];
            fft_in[k] = Complex(re, im);
        }
        
        // IFFT
        irfft(fft_in.data(), frame_out.data(), n_fft);
        
        // Store
        std::memcpy(&frames_time_domain[f * n_fft], frame_out.data(), n_fft * sizeof(float));
    }
    
    // Step 2: Overlap Add (Serial)
    std::vector<float> y(expected_len, 0.0f);
    std::vector<float> window_sum(expected_len, 0.0f);
    
    for (int f = 0; f < n_frames; ++f) {
        int start = f * hop_length;
        const float* frame_ptr = &frames_time_domain[f * n_fft];
        
        for (int i = 0; i < n_fft; ++i) {
            y[start + i] += frame_ptr[i] * window_padded[i];
            window_sum[start + i] += window_padded[i] * window_padded[i];
        }
    }
    
    // Normalize by window sum (avoid division by zero)
    for (int i = 0; i < expected_len; ++i) {
        if (window_sum[i] > 1e-8f) {
            y[i] /= window_sum[i];
        }
    }
    
    // Remove center padding and copy to output
    for (int i = 0; i < output_len; ++i) {
        if (pad_amount + i < expected_len) {
             output[i] = y[pad_amount + i];
        } else {
             output[i] = 0.0f;
        }
    }
}

} // namespace stft

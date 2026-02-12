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

#ifdef USE_OPENMP
#include <omp.h>
#endif

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
 * @param buffer Temporary buffer of size n (optional, handled internally if null)
 */
inline void rfft(const float* input, Complex* output, int n, std::vector<Complex>* buffer_ptr = nullptr) {
    // Copy to complex buffer
    // Use provided buffer to avoid allocation
    if (buffer_ptr) {
        if (buffer_ptr->size() < static_cast<size_t>(n)) buffer_ptr->resize(n);
        for (int i = 0; i < n; ++i) {
            (*buffer_ptr)[i] = Complex(input[i], 0.0f);
        }
        fft_radix2(buffer_ptr->data(), n, false);
        
        int n_out = n / 2 + 1;
        for (int i = 0; i < n_out; ++i) {
            output[i] = (*buffer_ptr)[i];
        }
    } else {
        std::vector<Complex> buffer(n);
        for (int i = 0; i < n; ++i) {
            buffer[i] = Complex(input[i], 0.0f);
        }
        
        fft_radix2(buffer.data(), n, false);
        
        int n_out = n / 2 + 1;
        for (int i = 0; i < n_out; ++i) {
            output[i] = buffer[i];
        }
    }
}

/**
 * Complex-to-real inverse FFT (irfft) matching torch.fft.irfft
 * @param input Complex input array of size n/2+1
 * @param output Real output array of size n
 * @param n_out Size of output (must be power of 2)
 * @param buffer Temporary buffer of size n_out (optional)
 */
inline void irfft(const Complex* input, float* output, int n_out, std::vector<Complex>* buffer_ptr = nullptr) {
    int n_freq = n_out / 2 + 1;
    
    if (buffer_ptr) {
        if (buffer_ptr->size() < static_cast<size_t>(n_out)) buffer_ptr->resize(n_out);
        for (int i = 0; i < n_freq; ++i) {
            (*buffer_ptr)[i] = input[i];
        }
         for (int i = n_freq; i < n_out; ++i) {
            (*buffer_ptr)[i] = std::conj((*buffer_ptr)[n_out - i]);
        }
        fft_radix2(buffer_ptr->data(), n_out, true);
        for (int i = 0; i < n_out; ++i) {
            output[i] = (*buffer_ptr)[i].real();
        }
    } else {
        std::vector<Complex> buffer(n_out);
        for (int i = 0; i < n_freq; ++i) {
            buffer[i] = input[i];
        }
        for (int i = n_freq; i < n_out; ++i) {
            buffer[i] = std::conj(buffer[n_out - i]);
        }
        
        fft_radix2(buffer.data(), n_out, true);
        
        for (int i = 0; i < n_out; ++i) {
            output[i] = buffer[i].real();
        }
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
    
    // Pre-allocate thread-local buffers
    int max_threads = 1;
    #ifdef USE_OPENMP
    max_threads = omp_get_max_threads();
    #endif
    
    std::vector<std::vector<float>> thread_frames(max_threads, std::vector<float>(n_fft));
    std::vector<std::vector<Complex>> thread_fft_outs(max_threads, std::vector<Complex>(n_freq));
    std::vector<std::vector<Complex>> thread_fft_buffers(max_threads, std::vector<Complex>(n_fft));
    
    // Process each frame
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int f = 0; f < n_frames; ++f) {
        int tid = 0;
        #ifdef USE_OPENMP
        tid = omp_get_thread_num();
        #endif
        
        std::vector<float>& frame = thread_frames[tid];
        
        int start = f * hop_length;
        
        for (int i = 0; i < n_fft; ++i) {
            frame[i] = padded[start + i] * window_padded[i];
        }
        
        // Compute FFT using pre-allocated buffers
        rfft(frame.data(), thread_fft_outs[tid].data(), n_fft, &thread_fft_buffers[tid]);
        
        // Store in output [n_freq, n_frames, 2] format
        for (int k = 0; k < n_freq; ++k) {
            // Note: Output layout is [Freq, Time, 2]
            output[(k * n_frames + f) * 2 + 0] = thread_fft_outs[tid][k].real();
            output[(k * n_frames + f) * 2 + 1] = thread_fft_outs[tid][k].imag();
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
    
    // Step 1: Compute all IFFTs in parallel
    std::vector<float> frames_time_domain(n_frames * n_fft);
    
    // Pre-allocate thread-local buffers
    int max_threads = 1;
    #ifdef USE_OPENMP
    max_threads = omp_get_max_threads();
    #endif
    
    std::vector<std::vector<Complex>> thread_fft_ins(max_threads, std::vector<Complex>(n_freq));
    std::vector<std::vector<float>> thread_frame_outs(max_threads, std::vector<float>(n_fft));
    std::vector<std::vector<Complex>> thread_fft_buffers(max_threads, std::vector<Complex>(n_fft));
    
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int f = 0; f < n_frames; ++f) {
        int tid = 0;
        #ifdef USE_OPENMP
        tid = omp_get_thread_num();
        #endif
        
        std::vector<Complex>& fft_in = thread_fft_ins[tid];
        std::vector<float>& frame_out = thread_frame_outs[tid];
        
        // Extract complex spectrum
        for (int k = 0; k < n_freq; ++k) {
            float re = stft_data[(k * n_frames + f) * 2 + 0];
            float im = stft_data[(k * n_frames + f) * 2 + 1];
            fft_in[k] = Complex(re, im);
        }
        
        // IFFT
        irfft(fft_in.data(), frame_out.data(), n_fft, &thread_fft_buffers[tid]);
        
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

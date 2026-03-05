#define DR_WAV_IMPLEMENTATION
#include "dr_libs/dr_wav.h"
#include "bs_roformer/audio.h"
#include <iostream>

AudioBuffer AudioFile::Load(const std::string& path) {
    AudioBuffer buffer;
    drwav_uint64 totalPCMFrames;

    // Get file info first
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), NULL)) {
        throw std::runtime_error("Failed to open audio file: " + path);
    }

    buffer.channels = wav.channels;
    buffer.sampleRate = wav.sampleRate;
    totalPCMFrames = wav.totalPCMFrameCount;
    buffer.samples = totalPCMFrames * buffer.channels;

    // Allocate directly in vector (avoid temporary buffer)
    buffer.data.resize(buffer.samples);
    drwav_read_pcm_frames_f32(&wav, totalPCMFrames, buffer.data.data());
    drwav_uninit(&wav);

    // Validation
    if (buffer.sampleRate != 44100) {
        std::cerr << "Warning: Input sample rate is " << buffer.sampleRate
                  << " Hz. Model expects 44100 Hz." << std::endl;
    }

    return buffer;
}

void AudioFile::Save(const std::string& path, const AudioBuffer& buffer) {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = buffer.channels;
    format.sampleRate = buffer.sampleRate;
    format.bitsPerSample = 32;
    
    drwav wav;
    if (!drwav_init_file_write(&wav, path.c_str(), &format, NULL)) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    
    drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, buffer.samples / buffer.channels, buffer.data.data());
    drwav_uninit(&wav);
    
    if (framesWritten != buffer.samples / buffer.channels) {
         throw std::runtime_error("Failed to write all samples to " + path);
    }
}

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile


def find_peaks(samples, sample_rate, output_path: str):
    freq, time, spectrogram = signal.spectrogram(samples, 
                                                 sample_rate, 
                                                 scaling='spectrum', 
                                                 window=('hann'))

    energy = np.sum(spectrogram, axis=0)
    peaks, _ = signal.find_peaks(energy, distance=1)

    plt.figure()
    plt.plot(time, energy)
    plt.plot(time[peaks], energy[peaks], "x")

    plt.xlabel('Время [c]')
    plt.ylabel('Энергия')
    plt.savefig(output_path)


def make_spectrogram(samples, sample_rate, output_path: str):
    freq, time, spectrogram = signal.spectrogram(samples, 
                                                 sample_rate, 
                                                 scaling='spectrum', 
                                                 window=('hann'))

    eps = np.finfo(float).eps
    spectrogram = np.maximum(spectrogram, eps)
    log_spectrogram = np.log10(spectrogram)
    
    plt.pcolormesh(time, freq, log_spectrogram, shading='gouraud')
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время  [c]')

    plt.savefig(output_path)


def noise_reduction_sagvol(samples, sample_rate, file_path: str, spec_path: str):
    sagvol_reduction = signal.savgol_filter(samples, 100, 5)

    wavfile.write(file_path, sample_rate, samples)
    make_spectrogram(sagvol_reduction, sample_rate, spec_path)

    return sagvol_reduction

def noise_reduction_butter(samples, sample_rate, file_path: str, spec_path: str):

    b, a = signal.butter(3, 4000 / sample_rate)
    butter_reduction = signal.filtfilt(b, a, samples)
    wavfile.write(file_path, sample_rate, butter_reduction)
    make_spectrogram(butter_reduction, sample_rate, spec_path)

    return butter_reduction


def main():
    sample_rate, samples = wavfile.read("input/sound.wav")

    peaks_path = "output/peaks.png"
    peaks_sagvol = "output/sagvol_peaks.png"
    peaks_butter = "output/butter_peaks.png"

    spec_path = "output/spectrogram.png"
    spec_sagvol = "output/sagvol_spectrogram.png"
    spec_butter = "output/butter_spectrogram.png"

    file_sagvol = "output/sagvol_sound.wav"
    file_butter = "output/butter_sound.wav"

    make_spectrogram(samples, sample_rate, spec_path)
    sagvol_reduction = noise_reduction_sagvol(samples, 
                                              sample_rate, 
                                              file_sagvol, 
                                              spec_sagvol
                                              )
    butter_reduction = noise_reduction_butter(samples, 
                           sample_rate,
                           file_butter,
                           spec_butter
                           )
    
    find_peaks(samples, sample_rate, peaks_path)
    find_peaks(sagvol_reduction, sample_rate, peaks_sagvol)
    find_peaks(butter_reduction, sample_rate, peaks_butter)

if __name__ == '__main__':
    main()
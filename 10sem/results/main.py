import numpy as np
import librosa
from glob import glob
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
from scipy.io import wavfile
import os



def make_spectrogram(samples, sample_rate, output_path: str):
    freq, time, spectrogram = signal.spectrogram(samples, 
                                                 sample_rate, 
                                                 scaling='spectrum', 
                                                 window=('hann'))

    log_spectrogram = np.log10(spectrogram)
    
    plt.pcolormesh(time, freq, log_spectrogram, shading='auto')
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время  [c]')

    plt.savefig(output_path)

    return freq, time, spectrogram

def get_frequences(input_path: str):
    samples, sample_rate = librosa.load(input_path, sr=None)
    db = librosa.amplitude_to_db(np.abs(librosa.stft(samples)), ref=np.max)

    freq = librosa.fft_frequencies(sr=sample_rate)
    mean_spec = np.mean(db, axis=1)

    min = np.argmax(mean_spec > -80)
    max = len(mean_spec) - np.argmax(mean_spec[::-1] > -80) - 1

    min_freq = freq[min]
    max_freq = freq[max]

    return max_freq, min_freq

def get_tembr_maintone(input_path: str):
    samples, sample_rate = librosa.load(input_path)
    chroma_stft = librosa.feature.chroma_stft(y=samples, sr=sample_rate)
    obertone = librosa.piptrack(y=samples, sr=sample_rate, S=chroma_stft)[0]
    max_obertone = np.argmax(obertone)

    return max_obertone

def get_formants(freq, time, spec):
    delta_t = int(0.1 * len(time))
    delta_freq = int(50 / (freq[1] - freq[0]))
    filtered = maximum_filter(spec, size=(delta_freq, delta_t))

    peaks_mask = (spec == filtered)
    peak_values = spec[peaks_mask]
    peak_frequencies = freq[peaks_mask.any(axis=1)]

    top_indices = np.argsort(peak_values)[-3:]
    top_frequencies = peak_frequencies[top_indices]

    return list(top_frequencies)

def process_sounds(input_path: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    spec_opath = os.path.join(output_path, os.path.splitext("spec_" + os.path.basename(input_path))[0] + ".png")
    title =os.path.basename(input_path).split(".")[0]
    file_opath = f"output/res_{title}.txt"

    with open(file_opath, 'w') as file:
        sample_rate, samples = wavfile.read(input_path)
        freq, time, spec = make_spectrogram(samples, sample_rate, spec_opath)
        max_freq, min_freq = get_frequences(input_path)
        formants = get_formants(freq, time, spec)
        maintone = get_tembr_maintone(input_path)
        file.write(f"{title}\n")
        file.write(f"Max frequency: {max_freq}\n")
        file.write(f"Min freq: {min_freq}\n")
        file.write(f"Maintone: {maintone}\n")
        file.write(f"Strongest formants: {formants}\n")


def main():
    process_sounds("input/a.wav")
    process_sounds("input/i.wav")
    process_sounds("input/gav.wav")



if __name__ == '__main__':
    main()
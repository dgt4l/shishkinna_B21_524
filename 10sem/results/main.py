import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile


def make_spectrogram(samples, sample_rate, output_path: str):
    freq, time, spectrogram = signal.spectrogram(samples, 
                                                 sample_rate, 
                                                 scaling='spectrum', 
                                                 window=('hann'))

    # eps = np.finfo(float).eps
    # spectrogram = np.maximum(spectrogram, eps)
    log_spectrogram = np.log10(spectrogram)
    
    plt.pcolormesh(time, freq, log_spectrogram, shading='auto')
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время  [c]')

    plt.savefig(output_path)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import torch
from PIL import Image

class DataTransformation():
    def __init__(self):
        self.sampling_rate = 125
        self.nfft = 32
        self.noverlap = 4

        self.plot_size = 128
    
    def generate_spectrogram(self, ecg_signal):
        nperseg = self.nfft

        ### Add padding ###
        # pad_length = self.nfft - (len(ecg_signal) % self.nfft)
        # if pad_length < self.nfft:
        #     ecg_signal = np.pad(ecg_signal, (0, pad_length), mode='constant')
        
        frequencies, times, Sxx = spectrogram(ecg_signal, fs=self.sampling_rate, window='blackman', nfft=self.nfft, noverlap=self.noverlap, nperseg=nperseg)

        return frequencies, times, Sxx

    def normalize_spectrogram(self, Sxx):
        # Normalize and reshape Sxx for CNN input
        Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Convert to dB
        Sxx_normalized = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db))  # Normalize
        Sxx_reshaped = Sxx_normalized[np.newaxis, :, :]  # Shape to (1, num_frequencies, num_times)

        return Sxx_reshaped
    
    def plot_spectrogram(self, frequencies, times, Sxx, plot=False):
        plt.figure(figsize=(10, 6)) # This resolution is too high for the CNN. This is only meant for observation.
        plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('ECG Signal Spectrogram')
        
        if plot:
            plt.show()

        plt.clf()
        plt.close()
    
    def get_spectrogram_plot(self, frequencies, times, Sxx, plot=False):
        fig, ax = plt.subplots(figsize=(self.plot_size/100, self.plot_size/100))
        c = ax.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud')

        # Turn off axes and ticks
        ax.axis('off')  # This removes the axes
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks

        # Draw the figure canvas
        fig.canvas.draw()

        ### Plot image ### 
        if plot:
            # Plot spectrogram before converting to array
            plt.show()

        # Convert plot to array
        rgb_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb_data = rgb_data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) # Size should be (height, width, 3)

        ### Plot spectrogram from the rgb_data. This is to ensure that the converted array matches the original plot ###
        if plot:
            plt.imshow(rgb_data)
            plt.axis('off')
            plt.show()

        plt.clf()
        plt.close()

        return rgb_data
    

    def transform(self, ecg_signal):
        frequencies, times, Sxx = self.generate_spectrogram(ecg_signal)
        rgb_data = self.get_spectrogram_plot(frequencies, times, Sxx)
        rgb_data = rgb_data.transpose((2, 0, 1)) # Put rgb channel in first dimension
        rgb_data = torch.tensor(rgb_data, dtype=torch.float32)
        return rgb_data
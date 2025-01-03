import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' to avoid graphical issues


# Importing libraries
import librosa  # For loading audio files and processing audio signals
import numpy as np  # For performing numerical computations like generating noise
import matplotlib.pyplot as plt  # For plotting graphical charts
import scipy.signal as signal  # For using signal processing functions like low-pass filters
import soundfile as sf  # For saving audio files
import sounddevice as sd  # For audio playback

# Function to apply a low-pass filter
# This filter removes high-frequency components above a specified cutoff frequency to reduce high-frequency noise
def low_pass_filter(audio, sr, cutoff=5000):
    # Calculate the Nyquist frequency, which is half the sampling rate
    nyquist = 0.5 * sr
    # Normalize the cutoff frequency relative to the Nyquist frequency
    normal_cutoff = cutoff / nyquist
    # Design a low-pass filter using scipy's butter function
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    # Apply the low-pass filter to the audio signal
    return signal.filtfilt(b, a, audio)

# Function to run the NLMS (Normalized Least Mean Squares) algorithm
# This is an adaptive filter algorithm that tries to remove noise from the signal
def nlms_filter(desired, input_signal, mu=0.001, M=128, epsilon=1e-6):
    N = len(desired)  # Number of samples
    w = np.zeros(M)  # Adaptive filter weights (initially set to zero)
    output_signal = np.zeros(N)  # Output signal with reduced noise
    error_signal = np.zeros(N)   # Error signal (difference between desired and output signal)

    # Loop to process the signal sample by sample
    for n in range(M, N):
        # Take a window of input samples of length M
        x = input_signal[n-M:n]
        # Calculate the output signal by dot product of weights and input window
        output_signal[n] = np.dot(w, x)
        # Calculate the error as the difference between desired and output signal
        error_signal[n] = desired[n] - output_signal[n]

        # Compute the normalized learning rate based on the input signal
        mu_n = mu / (np.dot(x, x) + epsilon)
        # Update the filter weights
        w = w + mu_n * error_signal[n] * x

    # Return the output signal and error signal
    return output_signal, error_signal

# Loading a sample audio signal from librosa example
audio, sr = librosa.load(librosa.example('trumpet'), sr=None)
# Adding noise to the audio signal with a specified noise level
noise_level = 0.1
noisy_audio = audio + noise_level * np.random.randn(len(audio))

# Preprocessing the noisy signal: applying a low-pass filter to reduce high-frequency noise
noisy_audio_filtered = low_pass_filter(noisy_audio, sr, cutoff=5000)

# Initial settings for the NLMS algorithm:
mu = 0.002  # Learning rate, affecting the convergence of the algorithm
M = 128     # Length of the adaptive filter, number of input samples used for the filter

# Running the NLMS algorithm to reduce noise in the audio signal
output_audio, error_audio = nlms_filter(audio, noisy_audio_filtered, mu=mu, M=M)

# Saving the processed output signal and error signal to audio files
sf.write('output_audio_nlms_improved.wav', output_audio, sr)
sf.write('error_audio_nlms_improved.wav', error_audio, sr)

# Playing the processed output audio using the sounddevice library
sd.play(output_audio, sr)
sd.wait()  # Wait for the audio playback to finish

# Displaying graphical plots of the different signals for comparison
plt.figure(figsize=(10, 6))

# Plotting the original audio signal
plt.subplot(3, 1, 1)
plt.plot(audio[:5000])
plt.title('Original Audio')

# Plotting the noisy audio signal
plt.subplot(3, 1, 2)
plt.plot(noisy_audio[:5000])
plt.title('Noisy Audio')

# Plotting the output audio signal after noise reduction with the NLMS algorithm
plt.subplot(3, 1, 3)
plt.plot(output_audio[:5000])
plt.title('Output Audio (Improved NLMS)')

plt.tight_layout()
plt.show()

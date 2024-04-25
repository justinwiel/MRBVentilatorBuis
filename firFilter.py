import numpy as np
import matplotlib.pyplot as plt

def fir_filter(signal, num_taps, cutoff_freq, sampling_freq):
    # Design FIR filter coefficients
    nyquist_freq = 0.5 * sampling_freq
    cutoff = cutoff_freq / nyquist_freq
    coefficients = np.sinc(2 * cutoff * (np.arange(num_taps) - (num_taps - 1) / 2.))
    
    # Apply Blackman window
    coefficients *= np.blackman(num_taps)
    
    # Normalize coefficients to have unity gain at DC
    coefficients /= np.sum(coefficients)
    
    # Apply filter using convolution
    filtered_signal = np.convolve(signal, coefficients, mode='same')
    
    return filtered_signal

def main():
    # Define the filter parameters
    num_taps = 300  # Number of filter taps
    cutoff_freq = 0.5  # Cutoff frequency of the filter (normalized frequency, 0.0 to 0.5)
    sampling_freq = 100  # Sampling frequency

    # Generate a signal to filter
    noise_intensity = 0.5
    t = np.linspace(0, 1, num=1000)  # Time vector
    signal = np.sin(2 * np.pi * 5 * t)  # Example signal: 5 Hz sine wave
    noise = noise_intensity* np.random.normal(0,1,1000)
    noisy_signal = noise +signal

    # Apply the FIR filter to the signal
    filtered_signal = fir_filter(noisy_signal, num_taps, cutoff_freq, sampling_freq)
    plt.figure(figsize=(10, 8))

    ### plot everything seperately 
    # Plot original signal
    plt.subplot(3, 1, 1)
    plt.plot(t, signal, label='Original Signal', color='b')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Original, Noisy, and Filtered Signals, separate')
    plt.legend(loc='lower left')
    plt.grid(True)

    # Plot noisy signal
    plt.subplot(3, 1, 2)
    plt.plot(t, noisy_signal, label='Noisy Signal', color='g')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend(loc='lower left')
    plt.grid(True)

    # Plot filtered signal
    plt.subplot(3, 1, 3)
    plt.plot(t, filtered_signal, label='Filtered Signal', color='r')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend(loc='lower left')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    # Plot everything together for comparison
    plt.plot(t, signal, label='Original Signal', color='b')



    plt.plot(t, filtered_signal, label='Filtered Signal', color='r')

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Original and Filtered Signals, together')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()
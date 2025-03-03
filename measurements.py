import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import csv
from datetime import datetime
from scipy.signal import welch, sosfilt, butter
from os.path import exists

# Updated import statement for tone schedule
from tone_sched_wwv import schedule

def extract_iq_data(wav_file_path):
    """
    Extracts I/Q data from a stereo .wav file where the two channels represent I/Q samples.

    Parameters:
    wav_file_path (str): Path to the .wav file.

    Returns:
    Tuple[np.ndarray, int]: Tuple containing the array of complex I/Q samples and the sample rate.
    """
    try:
        # Read the WAV file using soundfile
        data, samplerate = sf.read(wav_file_path, dtype='float32')

        # Check if the file contains stereo data (2 channels for I/Q)
        if data.ndim != 2 or data.shape[1] != 2:
            print(f"File {wav_file_path} does not contain stereo I/Q data. Skipping.")
            return None, None

        # Combine I and Q channels into a single complex array
        iq_data = data[:, 0] + 1j * data[:, 1]

        # Check if data was read successfully
        if iq_data.size == 0:
            print(f"No audio data found in file {wav_file_path}.")
            return None, None

        return iq_data, samplerate

    except Exception as e:
        print(f"Error reading WAV file {wav_file_path}: {e}")
        return None, None


def bandpass_filter(data, samplerate, lowcut, highcut):
    """
    Bandpass filter for isolating specific frequency ranges.
    """
    sos = butter(6, [lowcut, highcut], btype='bandpass', fs=samplerate, output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data


def measure_noise_floor(
    iq_data, samplerate, start_time, end_time, 
    lower_freq=1500, upper_freq=8000, tone_freq=None, exclusion_band=10
):
    """
    Measures the noise floor within the specified frequency range, excluding the tone frequency region.

    Parameters:
    - iq_data (np.ndarray): Complex I/Q data array.
    - samplerate (int): Sampling rate of the data (Hz).
    - start_time (float): Start time of the period (seconds).
    - end_time (float): End time of the period (seconds).
    - lower_freq (float): Lower bound of the frequency range for noise measurement (Hz).
    - upper_freq (float): Upper bound of the frequency range for noise measurement (Hz).
    - tone_freq (float): Tone frequency to exclude (Hz). If None, no exclusion is applied.
    - exclusion_band (float): Frequency range around the tone frequency to exclude (Hz).

    Returns:
    - float: Robust estimation of the noise floor (10th percentile of power spectrum).
    """
    try:
        # Extract the data corresponding to the time range
        start_sample = int(start_time * samplerate)
        end_sample = int(end_time * samplerate)
        noise_data = iq_data[start_sample:end_sample]
        if len(noise_data) == 0:
            return None

        # Perform FFT to find spectral components in the noise region
        fft_data = np.fft.fftshift(np.fft.fft(noise_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(noise_data), d=1 / samplerate))
        magnitude_spectrum = np.abs(fft_data) ** 2  # Convert to power

        # Select only frequencies within the desired range (1500–8000 Hz)
        valid_indices = np.where((freqs >= lower_freq) & (freqs <= upper_freq))[0]
        valid_freqs = freqs[valid_indices]
        valid_magnitude_spectrum = magnitude_spectrum[valid_indices]

        # Exclude bins near the tone frequency (if specified)
        if tone_freq is not None:
            exclusion_indices = np.where(
                (valid_freqs >= (tone_freq - exclusion_band)) & (valid_freqs <= (tone_freq + exclusion_band))
            )[0]
            valid_indices = np.setdiff1d(valid_indices, exclusion_indices)
            valid_magnitude_spectrum = magnitude_spectrum[valid_indices]

        # Use 10th percentile of the valid power spectrum to estimate noise floor
        return np.percentile(valid_magnitude_spectrum, 10) if len(valid_magnitude_spectrum) > 0 else None

    except Exception as e:
        print(f"Error measuring noise floor: {e}")
        return None


def measure_tone_power_and_freq_dev(iq_data, samplerate, tone_freq, start_time, end_time):
    """
    Measures the power and frequency deviation of the scheduled tone.
    """
    try:
        # Slice the data for the time range
        start_sample = int(start_time * samplerate)
        end_sample = int(end_time * samplerate)
        tone_data = iq_data[start_sample:end_sample]
        if len(tone_data) == 0:
            return None, None

        # Apply bandpass filtering around the tone frequency
        tone_data_filtered = bandpass_filter(tone_data, samplerate, tone_freq - 4, tone_freq + 4)

        # Perform FFT to analyze frequency content in the bandpass-filtered data
        fft_data = np.fft.fftshift(np.fft.fft(tone_data_filtered))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(tone_data_filtered), d=1 / samplerate))
        magnitude_spectrum = np.abs(fft_data)

        # Exclude low frequencies to ignore the carrier
        valid_indices = np.where((freqs >= (tone_freq - 4)) & (freqs <= (tone_freq + 4)))[0]
        freqs = freqs[valid_indices]
        magnitude_spectrum = magnitude_spectrum[valid_indices]

        # Find peak frequency 
        peak_index = np.argmax(magnitude_spectrum)
        peak_freq = freqs[peak_index]
        # calculate power
        tone_power = magnitude_spectrum[peak_index] ** 2
        # calculate frequency deviation
        freq_deviation = peak_freq - tone_freq  
        # A deviation greater than 4 Hz is considered too high because it indicates significant frequency instability.
        if abs(freq_deviation) > 4:
            print(f"Warning: Tone frequency deviation is too high: {freq_deviation} Hz")

        return tone_power, freq_deviation
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.error("Error measuring tone power and frequency deviation", exc_info=True)
        print(f"Error measuring tone power and frequency deviation: {e}")
        return None, None

def measure_tone_min_power_and_freq_dev(iq_data, samplerate, tone_freq, start_time, end_time):
    """
    Measures the minimum power in the passband and the frequency deviation of the scheduled tone.
    """
    try:
        # Slice the data for the time range
        start_sample = int(start_time * samplerate)
        end_sample = int(end_time * samplerate)
        tone_data = iq_data[start_sample:end_sample]
        if len(tone_data) == 0:
            return None, None

        # Apply bandpass filtering around the tone frequency
        tone_data_filtered = bandpass_filter(tone_data, samplerate, tone_freq - 4, tone_freq + 4)

        # Perform FFT to analyze frequency content in the bandpass-filtered data
        fft_data = np.fft.fftshift(np.fft.fft(tone_data_filtered))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(tone_data_filtered), d=1 / samplerate))
        magnitude_spectrum = np.abs(fft_data)

        # Exclude low frequencies to ignore the carrier
        valid_indices = np.where((freqs >= (tone_freq - 4)) & (freqs <= (tone_freq + 4)))[0]
        freqs = freqs[valid_indices]
        magnitude_spectrum = magnitude_spectrum[valid_indices]

        # Calculate the minimum power in the passband
        min_index = np.argmin(magnitude_spectrum)
        min_freq = freqs[min_index]
        tone_min_power = magnitude_spectrum[min_index] ** 2

        # Calculate frequency deviation (use the minimum frequency index for strict consistency)
        freq_deviation = min_freq - tone_freq

        # Optionally, check if the deviation suggests instability
        if abs(freq_deviation) > 4:
            print(f"Warning: Tone frequency deviation is too high: {freq_deviation} Hz")

        return tone_min_power, freq_deviation
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.ERROR)
        logging.error("Error measuring tone minimum power and frequency deviation", exc_info=True)
        print(f"Error measuring tone minimum power and frequency deviation: {e}")
        return None, None


def write_csv(filename, data):
    """
    Writes the measurement data to a CSV file.

    Parameters:
    filename (str): Path to the CSV file.
    data (list of dict): List of dictionaries containing measurement data.

    Returns:
    None
    """
    fieldnames = [
        'Time',
        'WWV Tone (Hz)',
        'WWV Power',
        'WWVH Tone (Hz)',
        'WWVH Power',
        'Noise Power',
        'WWV Freq Dev (Hz)',
        'WWVH Freq Dev (Hz)'
    ]
    try:
        # Check if the file exists to determine whether to write a header or append to an existing file
        file_exists = exists(filename)
        write_header = not file_exists

        # Write to the CSV file
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in data:
                writer.writerow(row)
    except Exception as e:
        print(f"Error writing to CSV file {filename}: {e}")

""" Plotting deferred for now...
def plot_snr(data, output_dir):
    # Helper function to parse and validate timestamps
    def parse_time(time_str):
        try:
            # Parse the timestamp in the expected format: 'YYYYMMDDTHHMMSSZ'
            dt = datetime.strptime(time_str, '%Y%m%dT%H%M%SZ')
            return dt
        except ValueError:
            # If the timestamp is not in the expected format, return None
            return None

    # Filter valid data
    valid_data = list()
    for entry in data:
        if 'Time' in entry and entry['WWV Power'] is not None and entry['WWVH Power'] is not None and entry['Noise Power'] is not None:
            dt = parse_time(entry['Time'])
            if dt:
                entry['Parsed Time'] = dt  # Add parsed datetime for later use
                valid_data.append(entry)

    if not valid_data:
        print("No valid data available for SNR plotting.")
        return

    # Extract time in minutes and SNR metrics
    time_in_minutes = [
        dt.hour * 60 + dt.minute for dt in (entry['Parsed Time'] for entry in valid_data)
    ]
    snr_wwv = [entry['WWV Power'] - entry['Noise Power'] for entry in valid_data]
    snr_wwvh = [entry['WWVH Power'] - entry['Noise Power'] for entry in valid_data]
    snr_ratio = [
        wwvh / wwv if wwv != 0 else 0 for wwvh, wwv in zip(snr_wwvh, snr_wwv)
    ]

    # Generate the plot
    plt.figure(figsize=(14, 6))
    plt.scatter(time_in_minutes, snr_wwvh, color='blue', label='WWVH SNR', alpha=0.6, s=10)
    plt.scatter(time_in_minutes, snr_wwv, color='red', label='WWV SNR', alpha=0.6, s=10)
    plt.scatter(time_in_minutes, snr_ratio, color='green', label='SNR Ratio (WWVH/WWV)', alpha=0.6, s=10)

    plt.xticks(
        range(0, 1440, 60),  # Matches 24 hour intervals (0-1380 minutes; 24 values)
        [f'{hour:02d}00' for hour in range(24)]  # Labels for each hour, total 24 labels
    )

    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('SNR')
    plt.title('SNR Comparison: WWV, WWVH, and Ratio')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the plot to a file
    output_file = f"{output_dir}/snr_plot.png"
    plt.savefig(output_file)
    print(f"SNR Plot saved as {output_file}")
    plt.close()

def plot_frequency_deviation_wwv(data, output_dir):
    def parse_time(time_str):
        try:
            dt = datetime.strptime(time_str, '%Y%m%dT%H%M%SZ')  # Updated format
            return dt
        except ValueError:
            return None

    # Filter valid data
    valid_data =  list()
    for entry in data:
        if 'Time' in entry and entry.get('WWV Frequency Deviation') is not None:
            dt = parse_time(entry['Time'])
            if dt:
                entry['Parsed Time'] = dt
                valid_data.append(entry)

    if not valid_data:
        print("No valid data available for WWV Frequency Deviation plotting.")
        return

    # Time in minutes and frequency deviation
    time_in_minutes = [
        dt.hour * 60 + dt.minute for dt in (entry['Parsed Time'] for entry in valid_data)
    ]
    wwv_frequency_deviation = [
        entry['WWV Frequency Deviation'] / 10.0 for entry in valid_data
    ]

    plt.figure(figsize=(14, 6))
    plt.scatter(
        time_in_minutes,
        wwv_frequency_deviation,
        color='blue',
        label='WWV Frequency Deviation (x10^-10)',
        alpha=0.6,
        s=10
    )

    plt.xticks(
        range(0, 1440, 60),  # Matches 24 hour intervals (0-1380 minutes; 24 values)
        [f'{hour:02d}00' for hour in range(24)]  # Labels for each hour, total 24 labels
    )

    
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Frequency Deviation (x10^-10)')
    plt.title('WWV Frequency Deviation Throughout the Day')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_file = f"{output_dir}/wwv_frequency_deviation_plot.png"
    plt.savefig(output_file)
    print(f"WWV Frequency Deviation Plot saved as {output_file}")
    plt.close()

def plot_frequency_deviation_wwvh(data, output_dir):
    def parse_time(time_str):
        try:
            dt = datetime.strptime(time_str, '%Y%m%dT%H%M%SZ')  # Updated format
            return dt
        except ValueError:
            return None

    # Filter valid data
    valid_data = list()
    for entry in data:
        if 'Time' in entry and entry.get('WWVH Frequency Deviation') is not None:
            dt = parse_time(entry['Time'])
            if dt:
                entry['Parsed Time'] = dt
                valid_data.append(entry)

    if not valid_data:
        print("No valid data available for WWVH Frequency Deviation plotting.")
        return

    # Time in minutes and frequency deviation
    time_in_minutes = [
        dt.hour * 60 + dt.minute for dt in (entry['Parsed Time'] for entry in valid_data)
    ]
    wwvh_frequency_deviation = [
        entry['WWVH Frequency Deviation'] / 10.0 for entry in valid_data
    ]

    plt.figure(figsize=(14, 6))
    plt.scatter(
        time_in_minutes,
        wwvh_frequency_deviation,
        color='red',
        label='WWVH Frequency Deviation (x10^-10)',
        alpha=0.6,
        s=10
    )
    plt.xticks(
        range(0, 1440, 60),  # Matches 24 hour intervals (0-1380 minutes; 24 values)
        [f'{hour:02d}00' for hour in range(24)]  # Labels for each hour, total 24 labels
    )
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Frequency Deviation (x10^-10)')
    plt.title('WWVH Frequency Deviation Throughout the Day')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_file = f"{output_dir}/wwvh_frequency_deviation_plot.png"
    plt.savefig(output_file)
    print(f"WWVH Frequency Deviation Plot saved as {output_file}")
    plt.close()

def plot_combined_frequency_deviation(data, output_dir):
    def parse_time(time_str):
        try:
            dt = datetime.strptime(time_str, '%Y%m%dT%H%M%SZ')  # Updated format
            return dt
        except ValueError:
            return None

    # Filter valid data
    valid_data = list()
    for entry in data:
        if (
            'Time' in entry and
            entry.get('WWV Frequency Deviation') is not None and
            entry.get('WWVH Frequency Deviation') is not None
        ):
            dt = parse_time(entry['Time'])
            if dt:
                entry['Parsed Time'] = dt
                valid_data.append(entry)

    if not valid_data:
        print("No valid data available for Combined Frequency Deviation plotting.")
        return

    # Extract time and deviations
    time_in_minutes = [
        dt.hour * 60 + dt.minute for dt in (entry['Parsed Time'] for entry in valid_data)
    ]
    wwv_frequency_deviation = [
        entry['WWV Frequency Deviation'] / 10.0 for entry in valid_data
    ]
    wwvh_frequency_deviation = [
        entry['WWVH Frequency Deviation'] / 10.0 for entry in valid_data
    ]

    plt.figure(figsize=(14, 6))
    plt.scatter(
        time_in_minutes,
        wwv_frequency_deviation,
        color='blue',
        label='WWV Frequency Deviation (x10^-10)',
        alpha=0.6,
        s=10
    )
    plt.scatter(
        time_in_minutes,
        wwvh_frequency_deviation,
        color='red',
        label='WWVH Frequency Deviation (x10^-10)',
        alpha=0.6,
        s=10
    )
    plt.xticks(
        range(0, 1440, 60),  # Matches 24 hour intervals (0-1380 minutes; 24 values)
        [f'{hour:02d}00' for hour in range(24)]  # Labels for each hour, total 24 labels
    )
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Frequency Deviation (x10^-10)')
    plt.title('Combined Frequency Deviations (WWV and WWVH)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_file = f"{output_dir}/combined_frequency_deviation_plot.png"
    plt.savefig(output_file)
    print(f"Combined Frequency Deviation Plot saved as {output_file}")
    plt.close()
"""

def process_files(directory):
    """
    Main processing function to read .wav files, analyze samples, and generate outputs.

    Parameters:
    directory (str): Path to the directory containing .wav files.

    Returns:
    None
    """
    try:
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            return
        files = sorted([f for f in os.listdir(directory) if f.endswith('.wav')])
        if not files:
            print(f"No .wav files found in directory: {directory}")
            return

        data = list()
        for file in files:
            file_path = os.path.join(directory, file)
            # Extract datetime from filename (e.g., '20250222T235900Z_10000000_iq.wav')
            datetime_str = file.split('_')[0]
            file_datetime = datetime.strptime(datetime_str, '%Y%m%dT%H%M%SZ')
            day_minute = file_datetime.hour * 60 + file_datetime.minute

            # Get expected tones from the schedule
            schedule_minute = day_minute % 60
            tone_info = schedule.get(schedule_minute, {'WWV': None, 'WWVH': None})
            wwv_tone = tone_info['WWV']
            wwvh_tone = tone_info['WWVH']

            # Process if at least one tone is scheduled
            if wwv_tone is not None or wwvh_tone is not None:
                iq_data, samplerate = extract_iq_data(file_path)
                if iq_data is None or len(iq_data) == 0:
                    print(f"Invalid or empty I/Q data from {file_path}. Skipping.")
                    continue

                print(f"Processing file: {file_path}")
                print(f"Extracted {len(iq_data)} samples at {samplerate} Hz.")

                # Initialize measurement values
                wwv_power, wwvh_power = None, None
                wwv_freq_dev, wwvh_freq_dev = None, None

                '''# Measure noise floor within 1300–3000 Hz range, excluding the tone frequency
                noise_floor = measure_noise_floor(
                    iq_data, samplerate, start_time=1, end_time=44, lower_freq=1300, upper_freq=3000, tone_freq=wwv_tone if wwv_tone else wwvh_tone, exclusion_band=10
                )'''

                # Bandpass filter and measure noise floor
                noise_freq = 2000  # Noise floor measurement frequency
                noise_filtered_data = bandpass_filter(iq_data, samplerate, noise_freq -400, noise_freq + 400)
                noise_floor, _ = measure_tone_min_power_and_freq_dev(noise_filtered_data, samplerate, noise_freq, start_time=1, end_time=44)

                # Bandpass filter and measure tone for WWV
                if wwv_tone is not None:
                    wwv_filtered_data = bandpass_filter(iq_data, samplerate, wwv_tone - 4, wwv_tone + 4)
                    wwv_power, wwv_freq_dev = measure_tone_power_and_freq_dev(
                        wwv_filtered_data, samplerate, wwv_tone, start_time=0, end_time=44
                    )

                # Bandpass filter and measure tone for WWVH
                if wwvh_tone is not None:
                    wwvh_filtered_data = bandpass_filter(iq_data, samplerate, wwvh_tone - 4, wwvh_tone + 4)
                    wwvh_power, wwvh_freq_dev = measure_tone_power_and_freq_dev(
                        wwvh_filtered_data, samplerate, wwvh_tone, start_time=0, end_time=44
                    )

                # Append results
                data.append({
                    'Time': datetime_str,
                    'WWV Tone (Hz)': wwv_tone,
                    'WWV Power': wwv_power,
                    'WWVH Tone (Hz)': wwvh_tone,
                    'WWVH Power': wwvh_power,
                    'Noise Power': noise_floor,
                    'WWV Freq Dev (Hz)': wwv_freq_dev,
                    'WWVH Freq Dev (Hz)': wwvh_freq_dev
                })
        print(f"Processed {len(data)} files!")

        # Write results to CSV
        output_dir = directory
        csv_filename = os.path.join(output_dir, 'measurement_results.csv')
        print( f"Writing measurement results to CSV file: {csv_filename}")
        write_csv(csv_filename, data)

        '''# Plot results
        print("Generating plot of SNR results")
        plot_snr(data, output_dir)

        print("Generating plot of frequency deviation results")
        plot_frequency_deviation_wwv(data, output_dir)
        plot_frequency_deviation_wwvh(data, output_dir)
        plot_combined_frequency_deviation(data, output_dir)
        '''

    except Exception as e:
        print(f"Error processing files: {e}")


# Replace this with the correct directory containing .wav files
if __name__ == '__main__':
    directory = '/Users/mjh/Sync/Jupyter/WWVH-latest/wav_files'
    process_files(directory)

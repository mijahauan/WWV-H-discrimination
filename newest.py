import csv
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import io
import soundfile as sf
import datetime
from tone_sched_wwv import schedule

def apply_bandpass_filter(file_path, tone_freq, samplerate):
    """
    Applies a bandpass filter to the audio file at the specified path.

    Parameters:
    file_path (str): The path to the audio file.
    tone_freq (float): The center frequency for the bandpass filter.
    samplerate (int): The sample rate of the audio file.

    Returns:
    bytes: The filtered audio data as a bytes object.
    """
    sox_cmd = ['sox', file_path, '-t', 'wav', '-']
    if tone_freq is not None:
        lowcut = tone_freq - 4
        highcut = tone_freq + 4
        sox_cmd.extend(['bandpass', str(tone_freq), str(highcut - lowcut)])
    try:
        filtered_output = subprocess.run(sox_cmd, stdout=subprocess.PIPE, check=True).stdout
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {file_path}: {e}")
        return None
    return filtered_output

def analyze_filtered_audio(filtered_output, start_sec, end_sec, samplerate):
    """
    Analyzes a segment of the filtered audio output to calculate its magnitude.

    Parameters:
    filtered_output (bytes): The filtered audio data.
    start_sec (int): The start second of the segment to analyze.
    end_sec (int): The end second of the segment to analyze.
    samplerate (int): The sample rate of the audio data.

    Returns:
    float: The calculated magnitude of the audio segment.
    """
    if not filtered_output:
        return None
    with io.BytesIO(filtered_output) as byte_stream:
        data, _ = sf.read(byte_stream)
    start_sample = int(start_sec * samplerate)
    end_sample = int(end_sec * samplerate)
    segment = data[start_sample:end_sample]
    magnitude = np.sqrt(np.mean(segment**2))
    return magnitude

def calculate_snr(tone_magnitude, noise_floor):
    """
    Calculates the Signal-to-Noise Ratio (SNR).

    Parameters:
    tone_magnitude (float): The magnitude of the tone signal.
    noise_floor (float): The magnitude of the noise floor.

    Returns:
    float: The calculated SNR, or None if the noise floor is zero.
    """
    return tone_magnitude / noise_floor if noise_floor > 0 else None

def snr_to_db(snr):
    """
    Converts SNR to decibels (dB).

    Parameters:
    snr (float): The SNR value to convert.

    Returns:
    float: The SNR value in decibels, or 0 for None or non-positive SNR.
    """
    return 20 * np.log10(snr) if snr is not None and snr > 0 else 0

def plot_snr(wwv_snr, wwvh_snr, ratio_snr, filename, date_str, frequency):
    """
    Plots the SNR data for WWV and WWVH and their ratio, saving the plot to a file.

    Parameters:
    wwv_snr (list): List of SNR values for WWV.
    wwvh_snr (list): List of SNR values for WWVH.
    ratio_snr (list): List of SNR ratio values between WWV and WWVH.
    filename (str): Filename to save the plot to.
    date_str (str): Date string for labeling the plot.
    frequency (str): Frequency string for labeling the plot.
    """
    plt.figure(figsize=(20, 6))
    minutes = np.arange(1440)
    wwv_snr_db = [snr_to_db(snr) if snr is not None else None for snr in wwv_snr]
    wwvh_snr_db = [snr_to_db(snr) if snr is not None else None for snr in wwvh_snr]
    ratio_snr_db = [snr_to_db(ratio) if ratio is not None else None for ratio in ratio_snr]

    plt.plot(minutes, wwvh_snr_db, color='blue', label='WWVH SNR', linestyle='-')
    plt.plot(minutes, wwv_snr_db, color='red', label='WWV SNR', linestyle='-')

    valid_ratios = [(minute, ratio) for minute, ratio in enumerate(ratio_snr_db) if ratio is not None]
    valid_minutes, valid_ratio_values = zip(*valid_ratios) if valid_ratios else ([], [])
    plt.bar(valid_minutes, valid_ratio_values, color='green', alpha=0.5, label='Ratio WWV/WWVH')

    plt.axhline(0, color='black', linewidth=0.8)
    plot_title = f'SNR and Ratio Comparison for {date_str}, Frequency: {frequency}'
    plt.title(plot_title)
    plt.xlabel('Minute')
    plt.ylabel('SNR (dB)')

    x_labels = [f'{int(minute/60):02d}{minute%60:02d}' for minute in minutes if minute % 60 == 0]
    x_positions = [minute for minute in minutes if minute % 60 == 0]
    plt.xticks(x_positions, x_labels, rotation=90)

    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main(directory):
    # Initialize
    samplerate = 16000  # Sample rate of the audio files
    wwv_snr = [None] * 1440
    wwvh_snr = [None] * 1440
    ratio_snr = [None] * 1440

    files = sorted([f for f in os.listdir(directory) if f.endswith('_iq.flac')])
    if not files:
        print("No files found in the directory.")
        return

    first_file = files[0]
    datetime_str = first_file.split('_')[0]
    frequency = first_file.split('_')[1]
    date_obj = datetime.datetime.strptime(datetime_str, '%Y%m%dT%H%M%SZ')
    date_str = date_obj.strftime('%Y-%m-%d')
    plot_filename = f'{date_str}_Frequency_{frequency}.png'
    data_filename = f'{date_str}_Frequency_{frequency}.csv'

    # Check if the CSV already exists to determine if we need to write headers
    write_headers = not os.path.exists(data_filename)

    with open(data_filename, 'a', newline='') as csvfile:
        fieldnames = ['DateTime', 'Tone_WWV', 'Tone_Magnitude_WWV', 'Noise_Floor_WWV', 'SNR_WWV', 'Tone_WWVH', 'Tone_Magnitude_WWVH', 'Noise_Floor_WWVH', 'SNR_WWVH', 'Ratio_SNR']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()

        for file_name in files:
            datetime_str = file_name.split('_')[0]
            file_datetime = datetime.datetime.strptime(datetime_str, '%Y%m%dT%H%M%SZ')
            day_minute = file_datetime.hour * 60 + file_datetime.minute

            schedule_minute = day_minute % 60

            tone_wwv = schedule[schedule_minute]['WWV']
            tone_wwvh = schedule[schedule_minute]['WWVH']

            file_path = os.path.join(directory, file_name)

            tone_magnitude_wwv = tone_magnitude_wwvh = None
            noise_floor_wwv = noise_floor_wwvh = None

            if tone_wwv is not None:
                filtered_output = apply_bandpass_filter(file_path, tone_wwv, samplerate)
                if filtered_output:
                    tone_magnitude_wwv = analyze_filtered_audio(filtered_output, 1, 44, samplerate)
                    noise_floor_wwv = analyze_filtered_audio(filtered_output, 45, 59, samplerate)
                    wwv_snr[day_minute] = calculate_snr(tone_magnitude_wwv, noise_floor_wwv)

            if tone_wwvh is not None:
                filtered_output = apply_bandpass_filter(file_path, tone_wwvh, samplerate)
                if filtered_output:
                    tone_magnitude_wwvh = analyze_filtered_audio(filtered_output, 1, 44, samplerate)
                    noise_floor_wwvh = analyze_filtered_audio(filtered_output, 45, 59, samplerate)
                    wwvh_snr[day_minute] = calculate_snr(tone_magnitude_wwvh, noise_floor_wwvh)

            if wwv_snr[day_minute] is not None and wwvh_snr[day_minute] is not None:
                ratio_snr[day_minute] = wwv_snr[day_minute] / wwvh_snr[day_minute]

            # Append data to CSV
            writer.writerow({
                'DateTime': datetime_str,
                'Tone_WWV': tone_wwv,
                'Tone_Magnitude_WWV': tone_magnitude_wwv if tone_magnitude_wwv is not None else 'None',
                'Noise_Floor_WWV': noise_floor_wwv if noise_floor_wwv is not None else 'None',
                'SNR_WWV': snr_to_db(wwv_snr[day_minute]) if wwv_snr[day_minute] is not None else 'None',
                'Tone_WWVH': tone_wwvh,
                'Tone_Magnitude_WWVH': tone_magnitude_wwvh if tone_magnitude_wwvh is not None else 'None',
                'Noise_Floor_WWVH': noise_floor_wwvh if noise_floor_wwvh is not None else 'None',
                'SNR_WWVH': snr_to_db(wwvh_snr[day_minute]) if wwvh_snr[day_minute] is not None else 'None',
                'Ratio_SNR': snr_to_db(ratio_snr[day_minute]) if ratio_snr[day_minute] is not None else 'None'
            })

    # Plot the SNR data
    plot_snr(wwv_snr, wwvh_snr, ratio_snr, plot_filename, date_str, frequency)

if __name__ == '__main__':
    directory = '/home/wsprdaemon/wsprdaemon/wav-archive.d/20240131/AC0G_EM38ww/KA9Q_0_WWV_IQ@S000123_131/WWV_10'
    main(directory)


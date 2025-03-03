import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


# Helper function to parse time into minutes since start of the day
def parse_time_in_minutes(dtg):
    try:
        dt = datetime.strptime(dtg, '%Y%m%dT%H%M%SZ')
        return dt.hour * 60 + dt.minute
    except ValueError:
        return None


# Plot the Signal to Noise Ratio (SNR) (WWV / WWVH)
def plot_snr(data, output_dir):
    """
    Plots the Signal-to-Noise Ratios (SNR) for WWV and WWVH signals, as well as the SNR ratio (WWV/WWVH).

    Parameters:
    - data (pandas.DataFrame): DataFrame containing measurement data.
    - output_dir (str): Output directory for saving the plot.

    Returns:
    - None
    """
    # Ensure required columns exist in the DataFrame
    required_columns = {'Time', 'WWV Power', 'WWVH Power', 'Noise Power'}
    if not required_columns.issubset(data.columns):
        print(f"DataFrame missing required columns: {required_columns - set(data.columns)}")
        return

    try:
        # Prepare lists for plotting
        times = []
        wwv_snrs = []
        wwvh_snrs =  []
        snr_ratios =  []      # Iterate through each row in the dataframe
        for _, row in data.iterrows():
            # Extract values from the current row
            wwv_power = row['WWV Power']
            wwvh_power = row['WWVH Power']
            noise_power = row['Noise Power']
            time_str = row['Time']

            # Calculate SNRs
            if pd.notna(wwv_power) and pd.notna(noise_power) and noise_power > 0:
                wwv_snr = 10 * np.log10(wwv_power / noise_power)
            else:
                wwv_snr = None

            if pd.notna(wwvh_power) and pd.notna(noise_power) and noise_power > 0:
                wwvh_snr = 10 * np.log10(wwvh_power / noise_power)
            else:
                wwvh_snr = None

            # Calculate SNR ratio 
            if wwv_snr is not None and wwvh_snr is not None:
                wwv_snr_linear = 10 ** (wwv_snr / 10)
                wwvh_snr_linear = 10 ** (wwvh_snr / 10)
                snr_ratio_linear = wwv_snr_linear / wwvh_snr_linear
                snr_ratio = 10 * np.log10(snr_ratio_linear)
            else:
                snr_ratio = None
            # Convert timestamp to minutes since the start of the day
            try:
                dt = datetime.strptime(time_str, '%Y%m%dT%H%M%SZ')
                time_in_minutes = dt.hour * 60 + dt.minute
            except:
                time_in_minutes = None

            # Append valid data for plotting
            if time_in_minutes is not None:
                times.append(time_in_minutes)
                wwv_snrs.append(wwv_snr)
                wwvh_snrs.append(wwvh_snr)
                snr_ratios.append(snr_ratio)

        # Ensure we have valid data for plotting
        if not times or not snr_ratios:
            print("No valid SNR data available for plotting.")
            return

        # Plot WWV and WWVH SNRs
        plt.figure(figsize=(14, 6))
        plt.scatter(times, wwv_snrs, color='red', alpha=0.6, s=2, label='WWV SNR')
        plt.scatter(times, wwvh_snrs, color='blue', alpha=0.6, s=2, label='WWVH SNR')

        # Plot SNR ratio
        
        plt.scatter(times, snr_ratios, color='green', alpha=0.6, s=2, label='SNR Ratio (WWV/WWVH)')
        plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, label='SNR Ratio = 1')

        # Configure time labels for the x-axis
        plt.xticks(
            range(0, 1440, 60),  # Marks every hour (0-1440 minutes)
            [f'{hour:02d}00' for hour in range(24)],  # Labels as '0000', '0100', etc.
        )
        plt.xlabel('Time of Day (Hours)')
        plt.ylabel('SNR (dB) / SNR Ratio')
        plt.title('Signal-to-Noise Ratios (SNR) and Ratio Between WWV and WWVH')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save the plot
        output_file = f"{output_dir}/snr_plot.png"
        plt.savefig(output_file)
        print(f"SNR Plot saved as {output_file}")
        plt.close()

    except Exception as e:
        print(f"Error in plot_snr: {e}")


# Plot the WWV Frequency Deviation
def plot_frequency_deviation_wwv(data, output_dir):
    valid_data = data.dropna(subset=['WWV Freq Dev (Hz)', 'Time'])

    plt.figure(figsize=(14, 6))
    plt.scatter(
        valid_data['Minutes'],
        valid_data['WWV Freq Dev (Hz)'],
        color='blue',
        alpha=0.6,
        s=1,
        label='WWV Frequency Deviation (Hz)'
    )
    plt.xticks(
        range(0, 1440, 60),
        [f'{hour:02d}00' for hour in range(24)]
    )
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Frequency Deviation (Hz)')
    plt.title('WWV Frequency Deviation Throughout the Day')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_file = f"{output_dir}/wwv_frequency_deviation_plot.png"
    plt.savefig(output_file)
    print(f"WWV Frequency Deviation Plot saved as {output_file}")
    plt.close()


# Plot the WWVH Frequency Deviation
def plot_frequency_deviation_wwvh(data, output_dir):
    valid_data = data.dropna(subset=['WWVH Freq Dev (Hz)', 'Time'])

    plt.figure(figsize=(14, 6))
    plt.scatter(
        valid_data['Minutes'],
        valid_data['WWVH Freq Dev (Hz)'],
        color='red',
        alpha=0.6,
        s=1,
        label='WWVH Frequency Deviation (Hz)'
    )
    plt.xticks(
        range(0, 1440, 60),
        [f'{hour:02d}00' for hour in range(24)]
    )
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Frequency Deviation (Hz)')
    plt.title('WWVH Frequency Deviation Throughout the Day')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_file = f"{output_dir}/wwvh_frequency_deviation_plot.png"
    plt.savefig(output_file)
    print(f"WWVH Frequency Deviation Plot saved as {output_file}")
    plt.close()


# Combined Plot of WWV and WWVH Frequency Deviation
def plot_combined_frequency_deviation(data, output_dir):
    valid_data = data.dropna(subset=['WWV Freq Dev (Hz)', 'WWVH Freq Dev (Hz)', 'Time'])

    plt.figure(figsize=(14, 6))
    plt.scatter(
        valid_data['Minutes'],
        valid_data['WWV Freq Dev (Hz)'],
        color='blue',
        alpha=0.6,
        s=3,
        label='WWV Frequency Deviation (Hz)'
    )
    plt.scatter(
        valid_data['Minutes'],
        valid_data['WWVH Freq Dev (Hz)'],
        color='red',
        alpha=0.6,
        s=1,
        label='WWVH Frequency Deviation (Hz)'
    )
    plt.xticks(
        range(0, 1440, 60),
        [f'{hour:02d}00' for hour in range(24)]
    )
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Frequency Deviation (Hz)')
    plt.title('Combined WWV and WWVH Frequency Deviations Throughout the Day')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_file = f"{output_dir}/combined_frequency_deviation_plot.png"
    plt.savefig(output_file)
    print(f"Combined Frequency Deviation Plot saved as {output_file}")
    plt.close()


# Main function
def main():
    # Change 'your_csv_file.csv' and 'output_directory' to your paths
    csv_file = '/Users/mjh/Sync/Jupyter/WWVH-latest/wav_files/measurement_results.csv'  # Path to the input CSV file
    output_dir = '/Users/mjh/Sync/Jupyter/WWVH-latest/wav_files/'  # Path to the output directory

    # Load data
    data = pd.read_csv(csv_file)

    # Parse Times into Minutes Since Start of the Day
    data['Minutes'] = data['Time'].apply(parse_time_in_minutes)

    # Generate Plots
    plot_snr(data, output_dir)
    plot_frequency_deviation_wwv(data, output_dir)
    plot_frequency_deviation_wwvh(data, output_dir)
    plot_combined_frequency_deviation(data, output_dir)

if __name__ == '__main__':
    main()

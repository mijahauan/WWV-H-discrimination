# WWV-H-discrimination

This python script uses the per-minute IQ files stored by wsprdaemon listening to 
frequencies 2.5, 5, 10, 15, 20 MHz on which WWV and WWVH broadcast simultaneously.  

Any analysis of receptions on these frequencies must address the fact that the data contain 
some combination of the two broadcasts.  

A published schedule of tones broadcast by WWV and WWVH enables an independent measurement of 
each station broadcasting on the same carrier frequency.
https://www.nist.gov/pml/time-and-frequency-division/time-distribution/radio-station-wwv/wwv-and-wwvh-digital-time-code

wsprdaemon (https://github.com/rrobinett/wsprdaemon) stores 1440 1-minute wv (WavPack) files in a 
24-hour UTC day centered on the carrier.  This script reads each file and applies a filter on 
440, 500, and 600 Hz.  If either station (or both) broadcasts a tone in the given minute, it does 
so from second 1 through 44.  The remainder of the minute, seconds 45-59, have no tone so this 
period provides a window to measure the noise floor.  

The script measures the magnitude of the tone and noise periods and calculates the SNR.  If both 
stations sound a tone in the same minute, it calculates a ratio of the two SNRs.  It also 
measures the deviation between the expected tone frequency and the received tone frequency.

This script produces the following (examples above):
1) a PNG of the WWV SNR, the WWVH SNR, and the ratio between the two.
2) a PNG of the frequency deviation for each of WWV and WWVH.
2) a CSV listing the WWV tone frequency, magnitude, and noise;
   the WWVH tone frequency, magnitude, and noise;
   and the frequency deviation for each.

One can calculate the SNR for each station and the ratio of two with that informaton.

It requires the 'tone_sched_wwv.py' file in the same directory as the script to provide the 
schedule of expected tones.  One defines the directory where the 1440 .wv files reside just
before invoking the main() function. You will have to ensure the libraries the script imports 
are available to your python3 interpreter.

I used an RX888 MKII 16bit SDR Receiver Radio LTC2208 ADC radio managed by 
ka9q-radio (https://github.com/ka9q/ka9q-radio)
in Ubuntu 22.04 on a LattePanda Sigma (https://www.lattepanda.com/lattepanda-sigma).

[](./Screenshot%202025-02-28%20at%2013.59.59.png)
[](./2025-02-22_Frequency_10000000_Deviation_WWV.png)
[](./2025-02-22_Frequency_10000000_Deviation_WWVH.png)
[](./2025-02-22_Frequency_10000000_SNR.png)
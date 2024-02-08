# WWV-H-discrimination

This python script relies on the per-minute IQ files stored by wsprdaemon listening to WWV/H 
frequencies 2.5, 5, 10, 15, 20 MHz.  

A published schedule of tones broadcast by WWV and WWVH enables an independent measurement of 
each station broadcasting on the same carrier frequency.
https://www.nist.gov/pml/time-and-frequency-division/time-distribution/radio-station-wwv/wwv-and-wwvh-digital-time-code

wsprdaemon (https://github.com/rrobinett/wsprdaemon) stores 1440 1-minute FLAC files in a 
24-hour UTC day centered on the carrier.  This script reads each file and applies a filter on 
440, 500, and 600 Hz.  If either station (or both) broadcasts a tone in the given minute, it does 
so from second 1 through 44.  The remainder of the minute, seconds 45-59, have no tone so this 
period provides a window to measure the noise floor.  The script measures the magnitude of the 
tone and noise periods and calculates the SNR.  If both stations sound a tone in the same 
minute, it calculates a ratio of the two SNRs.  

This script produces the following (examples above):
1) a PNG of the WWV SNR, the WWVH SNR, and the ratio between the two.
2) a CSV listing the WWV tone frequency, magnitude, and SNR;
   the WWVH tone frequency, magnitude, and SNR;
   and the ratio of the two SNRs.  

It requires the 'tone_sched_wwv.py' file in the same directory as the script to provide the 
schedule of expected tones.  One defines the directory where the 1440 FLAC files reside just
before invoking the main() function. You will have to ensure the libraries the script imports 
are available to your python3 interpreter.

I used an RX888 MKII 16bit SDR Receiver Radio LTC2208 ADC radio managed by 
ka9q-radio (https://github.com/ka9q/ka9q-radio)
in Ubuntu 22.04 on a LattePanda Sigma (https://www.lattepanda.com/lattepanda-sigma).

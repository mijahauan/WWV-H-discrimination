# WWV-H-discrimination

This python script relies on the per-minute IQ files stored by wsprdaemon listening to WWV/H 
frequencies 2.5, 5, 10, 15, 20 MHz.  A published schedule of tones broadcast by WWV and WWVH
enables an independent measurement of each station broadcasting on the same carrier frequency.
https://www.nist.gov/pml/time-and-frequency-division/time-distribution/radio-station-wwv/wwv-and-wwvh-digital-time-code

wsprdaemon stores 1440 1-minute FLAC files centered on the carrier.  This script reads each
file and applies a filter on 440, 500, and 600 Hz.  Each station broadcasts a tone during 
schedule minutes that differs from the other station.  
https://github.com/rrobinett/wsprdaemon

This script produces two products:
1) a PNG of the WWV SNR, the WWVH SNR, and the ratio between the two.
2) a CSV listing the WWV tone frequency, magnitude, and SNR,
   the WWVH tone freuquency, magnitude, and SNR
   and the ratio of the two SNRs.  

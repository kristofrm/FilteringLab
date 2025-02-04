#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lab3_script.py
Creates hearing aid filter via an equal loudness curve that is based on the relative loudness of different frequency notes
Kristof Rohaly-Medved and Shea Moroney
Adapted code from ChatGPT at times as needed
"""

import numpy as np
import lab3_module as l3m
from matplotlib import pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

#%% Part 1: Determine Volumes of Equal Loudness

# Estimated hearing range
print("Estimated hearing range: -4 to 5")

# Get 10 notes based on octaves up and volume relative to the 1 kHz base signal
all_octaves_up = np.linspace(-4, 5, 10)
all_volumes = np.array([1.5, 1.25, 1.25, 1, 1, 1, 0.5, 0.25,  0.1, 0.3])

# Make the 10 notes into a chord
chord = l3m.make_chord(all_octaves_up, all_volumes)
#sd.play(chord)
sd.wait()

# Reflection on loudness of notes in chord
print("")
print("All the notes are blending together and sound as though they are equally loud despite not necessarily having"
     " equal physical volume.")

#%% Part 2: Plot Your Equal Loudness Curve

# Base frequency = 1 kHz. Get actual frequencies of the 10 notes in the chord based on octaves up from 1 kHz signal
base_frequency = 1000
chord_frequencies = base_frequency * (2**all_octaves_up)

# Convert the volumes into decibels using module function
all_volumes_db = l3m.convert_to_db(all_volumes)

# Plot the 10 notes with frequency on the x-axis and the decibel volumes on the y-axis
plt.figure(1, figsize=(12, 8), clear=True)
plt.subplot(3,1,1)
plt.scatter(chord_frequencies, all_volumes_db)
plt.title('Test Points on Equal Loudness Curve')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.xscale('log')
plt.grid()

# Get the FFT of the chord from part 1, the corresponding frequencies, and get the decibel power of the chord
fs = 44100
fft_result = np.fft.rfft(chord)
f_fft = np.fft.rfftfreq(len(chord), 1/fs)
chord_db = l3m.convert_to_db(fft_result)

# Plot the chord in the frequency domain with frequency on the x-axis and the decibel power in the y-axis
plt.subplot(3,1,2)
plt.plot(f_fft, chord_db)
plt.xscale('log')
plt.title('FFT of Equal Loudness Chord')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.grid()

# Reflection on representation of chord in frequency domain with relative volumes shown
print('')
print('In this plot, each spike is one of the notes in the chord where the height of the peak in decibels is the relative volume of that note. '
      'As the plot shows, the relative volume of each note is approximately the same for the given frequency range. '
      'This differs from the idealized version, as that plot only accounts for the '
      'specific notes in the chord array whereas the this plot also shows any artifacts '
      'or noise that might be present in the signal')

# Interpolate the corresponding frequencies for the known volumes to get relationship between frequency and volume based on equal loudness
volume_fft = np.interp(f_fft, chord_frequencies, all_volumes)
volume_fft_db = l3m.convert_to_db(volume_fft)

# Plot the decibel power of the new equal loudness frequency-volume relationship
plt.subplot(3,1,3)
plt.plot(f_fft, volume_fft_db)
plt.xscale('log')
plt.title('Interpolated Equal Loudness Curve')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.grid()
plt.tight_layout()

# Reflection on curve compared to literature
print('')
print('My equal loudness curve does resemble the equal loudness curve from the literature in its general shape, but it also differs '
      'slightly. My curve is more gradual in its decline in volume as frequency increases and does not have the same bump that comes '
      'after 1 kHz. The tail end of my curve also does not increase quite as much. I believe this is because of both differences in my own '
      'hearing and the relatively poor quality of my audio equipment. When testing the octaves, it was often difficult to compare the loudness '
      'of the differing pitches, partly because my speakers were not great and would add more distortion at extreme low and high frequencies. '
      'My own hearing also contributes to this difference because my curve is based specifically on my (likely damaged) hearing and not a professionals.')

#%% Part 3: Create a Hearing Aid Filter

# Create new figure with 1 row x 2 columns
plt.figure(2, figsize=(12, 8), clear=True)

# Plot interpolated equal loudness curve from part 2 on second subplot
plt.subplot(1,2,2)
plt.plot(f_fft, volume_fft_db)
plt.xscale('log')
plt.title('Interpolated Equal Loudness Curve')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.grid()

# Reflection on type of filter that curve most closely resembles
print('')
print('This filter acts mostly as a band stop filter in that based on my hearing, lower frequency notes did not need to be made '
      'louder but higher frequency notes needed to be made quieter except for very high frequencies to bring the relative volumes together. '
      'In other words, the filter primarily attenuated higher frequencies up to a certain point where they were then amplified/restored again. ')

# Get impulse response function via inverse FFT from interpolated equal loudness curve in the frequency domain
impulse_response = np.fft.ifft(volume_fft)
# Shift the impulse response to be centered around 0
impulse_response_shifted = np.fft.fftshift(impulse_response)

# Set crop duration to 0.1 s and calculate the number of samples to be included in the signal to the left and the right of time = 0
crop_duration = 0.1
samples_to_crop_per_side = int(crop_duration * fs) // 2

# Find the center index around which the impulse response is centered
center_index = len(impulse_response_shifted) // 2

# Crop the impulse response by getting 0.05 s of data to the left and right of the center index. Get corresponding 1 s time
impulse_response_cropped = impulse_response_shifted[center_index-samples_to_crop_per_side:center_index + samples_to_crop_per_side]
time = np.arange(-crop_duration/2, crop_duration/2, 1/fs)

# Plot the cropped impulse response in the time domain
plt.subplot(1,2,1)
plt.plot(time, impulse_response_cropped)
plt.title('Impulse Response of Equal Loudness Filter')
plt.xlabel('Time (s)')
plt.ylabel('h(t)')
plt.grid()

# Create chord of 2 new notes based on how many octaves they differ from the base 1 kHz signal, but with the same volume as the 1 kHz
two_notes_octaves_up = np.array([-2.5, 2.5], dtype = float)
two_notes_volume_up = np.array([2, 2])
two_notes_chord = l3m.make_chord(two_notes_octaves_up, two_notes_volume_up)
#sd.play(two_notes_chord)
sd.wait()

# Reflection on relative loudness of the 2 notes
print("")
print('The higher note sounds louder. This makes sense with the equal loudness '
      'curve the curve seeks to slightly raise the loudness of lower notes and slightly dampen the loudness of higher notes')

# Get chord into frequency domain
two_notes_chord_frequency_domain = np.fft.rfft(two_notes_chord)

# Convolve chord with filter by multiplying in frequency domain, then convert back to time domain
two_notes_chord_frequency_domain_filtered = two_notes_chord_frequency_domain * volume_fft
chord_filtered = np.fft.ifft(two_notes_chord_frequency_domain_filtered)
chord_filtered = np.real(chord_filtered)

# play chord
#sd.play(chord_filtered)
sd.wait()

# Reflection on filtering the 2 new notes
print('')
print('The filter has brought the two notes to roughly the same loudness. This is because it dampened the higher note while slightly raising the lower note,'
      'thus bringing the two closer together.')
print("")

#%% Part 4: Test Your Filter on Speech

# Read in data for speech file
fs_test, speech = wavfile.read('test123.wav')
if fs == fs_test:
    print('Sampling rates match')
else:
    print('Sampling rates do not match')

#sd.play(speech)
sd.wait()

# Convolve speech with filter
speech_filtered = np.convolve(speech, np.abs(impulse_response_cropped), mode='same')
speech_filtered = np.real(speech_filtered)

# play test file
#sd.play(speech_filtered/10000)
sd.wait()

# Reflection on filtered test file sound
print("")
print('The filtered test file sounds much louder than the original. '
      'It had to be scaled down by 10000 because it was too loud.')

# Get time for plotting unfiltered speech / time = len(speech)/fs
time = np.arange(0, len(speech)/fs, 1/fs)

# Plot the original and filtered speech in the time domain
plt.figure(3, figsize=(12, 8), clear=True)
plt.subplot(1,2,1)
plt.plot(time, speech, label = 'Original')
plt.plot(time, speech_filtered, label = 'Filtered', alpha = 0.5, color = 'red')
plt.title('Original and Filtered Speech Signal\nin Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Sound Amplitude (A.U.)')
plt.grid()
plt.legend()

# Reflection on graph of filtered vs original test signal
print("")
print('The filtered version has a higher amplitude than the original signal. '
      'This is due to the filtered version correcting for the percieved volume '
      'of different frequencies, whether they need to be amplified or reduced. For the most part,'
      'the filter made lower frequencies a bit louder and higher frequencies a bit quieter as expected'
      'for a filter that is essentially a band stop')

# Get frequency domain representation of original and filtered speech
speech_frequency_domain = np.fft.rfft(speech)
speech_filtered_frequency_domain = np.fft.rfft(speech_filtered)
speech_frequencies = np.fft.rfftfreq(len(speech), 1/fs)

# Convert frequency to decibels
speech_decibels = l3m.convert_to_db(speech_frequency_domain)
speech_filterd_decibels = l3m.convert_to_db(speech_filtered_frequency_domain)

# Plot the original and filtered speech in the frequency domain
plt.subplot(1,2,2)
plt.plot(speech_frequencies, speech_decibels, label = 'Original')
plt.plot(speech_frequencies, speech_filterd_decibels, label = 'Filtered', alpha = 0.5, color = 'red')
plt.title('Original and Filtered Speech Signal\nin Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.xscale('log')
plt.grid()
plt.legend()

# Reflection on how filter has modified frequency content
print("")
print('Our filter uses the equal-loudness curve created above to modify the '
      'amplitudes of different frequencies in the signal. This matches our '
      'expectations of the filter lowering the percieved volume of higher '
      'frequencies and slightly amplifying the percieved volume of lower frequencies.')

#%%
# Saving the figures
plt.figure(1)
plt.savefig('determining_equal_loudness_curve.pdf')
plt.figure(2)
plt.savefig('interpolated_equal_loudness_filter.pdf')
plt.figure(3)
plt.savefig('filter_applied_to_speech.pdf')

# Reflection on filter for people with high frequency hearing trouble
print("")
print('A filter designed for someone with a type of hearing loss where they '
     'have difficulty hearing high frequencies would differ from our filter as '
     'it would amplify higher frequencies compared to our filter reducing higher '
     'frequencies. This means it would most likely be a high-pass filter. ' 
     'High notes would sound louder in this filter because higher frequencies '
     'would be enhanced in order to improve their ability to hear those higher '
     'frequencies.')










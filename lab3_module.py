#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lab3_module.py
Module for lab3_script with functions comparing the loudness of 2 notes, making a chord of notes, and converting an FFT to decibels
Kristof Rohaly-Medved and Shea Moroney
Adapted code from ChatGPT at times as needed
"""

import numpy as np
import sounddevice as sd

#%% Part 1: Determine Volumes of Equal Loudness

def test_octave(duration, octaves_up, volume):
    """
    
    Function that builds a 1 kHz tone followed by a note in a different octave with different volume relative to the 1 kHz tone

    Parameters
    ----------
    duration : int
        Time in seconds to play the notes for
        
    octaves_up : float
        The number of octaves up the second note is in relation to the 1 kHz base signal
        
    volume : int
        The amplitude modifier with which to multiply the second note, thus representing the volume of the second note

    Returns
    -------
    None.
        
    """
    
    fs = 44100 # Hz
    # used ChatGBT to get np.linespace() bc np.arange() was not working for me
    time =  np.linspace(0, duration, int(fs * duration), endpoint=False)
    f = 1000 # Hz
    cos_wave = np.cos(2* np.pi * f * time)
    cos_wave_scaled = volume * np.cos(2* np.pi * f * (2 ** octaves_up) * time)
    
    concatenated_waves = np.concatenate((cos_wave, cos_wave_scaled))
    
    sd.play(concatenated_waves, samplerate=fs)
    sd.wait()

    
def make_chord(all_octaves_up, all_volumes):
    """
    Function to make chord from all octave and volume pairs

    Parameters
    ----------
   all_octaves_up : 1D array of floats size (n, 1)
       Representation of all notes in 10 evenly-spaced octaves
       
       
    all_volumes : 1D array of floats size (n, 1)
        Representation of all volumes corresponding to the notes in each octave to 
        make all note equal to the volume of the original note (0)

    Returns
    -------
    chord : 1D array of floats size (fs, 1)
        1-second chord overlapping all notes at equal perceived loudness 
        
    """
    fs = 44100 # Hz
    duration = 1 # seconds
    base_frequency = 1000 # Hz
    time = np.arange(0, duration,1/fs)
    chord = np.zeros(len(time))
    
    waves_count = len(all_octaves_up)
    
    for wave_index in range(waves_count):
        cos_frequency_2 = base_frequency*(2**all_octaves_up[wave_index])
        cos_wave_2 = np.cos(2*np.pi*cos_frequency_2*time)
        chord += all_volumes[wave_index]*cos_wave_2/waves_count
        
    return chord
                                     
                                    
#%% Part 2: Plot Your Equal Loudness Curve


    
def convert_to_db(fft_result):
    """
    
    Function to convert frequencies to their normalized power in decibels

    Parameters
    ----------
    fft_result : 1D array of complex number frequencies size (n, 1)
        Representation of a signal in frequency space

    Returns
    -------
    power_db : 1D array of floats size (n, 1)
        Calculated power of a frequency space representation
        
    """
    
    fft_result_abs = np.abs(fft_result) # get absolute value
    power = fft_result_abs**2 # square to get power
    power_normalized = power/np.max(power) # normalize by dividing by max value
    power_db = 10*np.log10(power_normalized) # convert power to decibels
    
    return power_db






    
    
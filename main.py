#!/usr/bin/env python

# ==========================================
# Dismantler
# by Alex Baker
# iamalexbaker@gmail.com
# The University of Huddersfield
# Modfied: 02/02/14
#
# Cuts out randomly selected events from a
# source audio file and groups them according
# to randomly selected spectral features,
# then creates a new audio file with some
# chance-determined structure from these events.
# ==========================================

import sys, os, wave, random, struct
import numpy as np
from scipy.io import wavfile as wav
import scipy.signal as sig
from scipy.fftpack import rfft
from scipy import log10
from scipy.cluster.vq import kmeans2 as kmeans
import matplotlib.pyplot as plt

# ---------------
# User Variables
# These are here for you to change.
# There are no limits though, so certain
# values may crash everything!
# ---------------
sourceaudio = 'Coalesce_mono.wav' # note: this must be a mono, uncompressed wav file! In theory, any sample rate should work but only 44.1khz has been tested.
outputfile = 'hello.wav'
tracks = 4 # This is the number of tracks the event types are split into
num_events = 40 # This is the total number of events that will be extracted from the source audio
length_sec = 180 # The length in seconds of audio you want to generate
gap_sec = 0.2 # The gap in seconds that will be skipped every time an event isn't generated
grain_min = 4000 # The minimum length of a generated event (in samples)
grain_max = 160000 # The maximum length of a generated event
norm_level = 0.85 # The level that the output audio is normalised to (1.0 is 0dB)
density = 16.0 # The maximum density of grains - concretely, this is the maximum chance of generating a grain every gap_sec seconds (will be divided by the number of tracks)
num_features = 30 # Number of features to use for clustering - in theory, a higher number should cause events to be grouped more accurately
# ---------------

# Open wav
print "Opening "+sourceaudio+".."
ifile = wave.open(sourceaudio,'r')
iaudio = ifile.readframes(-1)
iaudio = np.fromstring(iaudio, 'Int16')
sr = ifile.getframerate()
ifile.close()
numsamples = len(iaudio)
faudio = iaudio.astype('Float16')/32767.

# Variables

length_sam = length_sec*sr
buffers = np.zeros((tracks*2,length_sam))
pos = np.zeros(tracks)
density_max = density/tracks
gap_sam = int(gap_sec*sr)
grain_mid = (grain_max-grain_min)/2

# ---------------
# add_event
# Adds a generated event to the appropriate buffers (channels by track)
# ---------------
def add_event(buf, ev_time, ev_start, ev_length, amp=0.5, pan=0.5):
    global buffers, length_sam, pos
    # If the event extends past the specified end time, extend all buffers to match it
    if ev_time+ev_length>length_sam:
        buffers = np.concatenate((buffers,np.zeros((tracks*2,(ev_time+ev_length)-length_sam))),axis=1)
        length_sam = ev_time+ev_length
    # Make sure the region of audio taken from the source track doesn't extend past the end of it
    ev_end = np.minimum(ev_start+ev_length,len(faudio))
    # Generate a Hanning window
    window = sig.hann(ev_end-ev_start)
    # Add event to the left channel
    buffers[buf,ev_time:ev_time+ev_length] = buffers[buf,ev_time:ev_time+ev_length]+(faudio[ev_start:ev_end]*amp*(1-pan)*window)
    # Add event to the right channel
    buffers[buf+tracks,ev_time:ev_time+ev_length] = buffers[buf+tracks,ev_time:ev_time+ev_length]+(faudio[ev_start:ev_end]*amp*pan*window)
    # Set the new position for this track
    pos[buf] = ev_time + ev_length
    
    
# ---------------
# normalise
# Normalises the given stereo audio track to the maximum value defined by the user
# ---------------
def normalise(stereo):
    peak = np.amax(stereo)
    print "Peak level is %.4f, normalising to %.2f.." % (peak,norm_level)
    return [stereo[0]*(norm_level/peak),stereo[1]*(norm_level/peak)]
        
# ---------------
# mixdown
# Mixes every buffer together to make the finished audio file
# and converts it to integer format
# ---------------
def mixdown():
    print "Mixing down to stereo.."
    # Sum and normalise all of the left and right channel buffers, respectively
    left = np.sum(buffers[0:tracks],0)
    right = np.sum(buffers[tracks:tracks*2],0)
    [left_n, right_n] = normalise([left,right])
    # Convert to 16 bit integer format
    mix = [(left_n*32767).astype('Int16'),(right_n*32767).astype('Int16')]
    # Transpose the matrix (I work with it oriented the other way around)
    return np.transpose(mix)

# ---------------
# export_tracks
# This is a debugging function used to export each individual track.
# If a track has no events on it, this will crash.
# ---------------
def export_tracks(_plot):
    print "Exporting individual tracks.."
    # For each track..
    for i in range(0,tracks):
        # Normalise left and right
        [left_n, right_n] = normalise([buffers[i],buffers[i+tracks]])
        # Convert to integer and transpose
        tr = np.transpose([(left_n*32767).astype('Int16'),(right_n*32767).astype('Int16')])
        print "Exporting track %d.." % (i+1)
        # Write the au8dio file
        wav.write('track_%d.wav'%(i+1),44100,tr)
        # Optional functionality that creates a plot of each track as it is exported
        if _plot:
            plot_audio(i+2,tr,sr,'Track %d'%(i+1))
            plt.show()

# ---------------
# filltrack
# Fills the given track with events
# ---------------
def filltrack(track):
    global pos
    # _nevents is the running total, just for display purposes
    _nevents = 0
    # Set the initial chance
    # This is equal to the density specified by the user (/4), multiplied by the track's own structure function added to the overall structure function 
    chance = density_max*((structure(pos[track]/sr,structures[track])+structure(pos[track]/sr,structures[tracks]))/2)
    # As long as the position hasn't reached the end of the track, keep trying to generate events
    while pos[track]<length_sam-1:
        # Try to generate an event with the chance set above
        if np.random.uniform(0.,1.,)<=chance:
            # Add a random event from the track's corresponding group, at the current position, with a random length between grain_min and grain_max,
            # with a random pan position and an amplitude defined partially by the overall structure with some noise
            add_event(track,pos[track],event_groups[track][np.random.randint(0,len(event_groups[track]))],np.minimum(np.random.randint(grain_min,grain_max),len(faudio)-grain_min),np.random.uniform(0.,0.05)+0.15*structure(pos[track]/sr,structures[tracks]),np.random.uniform(0.,1.))
            # Increment the current event count
            _nevents += 1
            # Calculate the chance in the same way specified above
            chance = density_max*((structure(pos[track]/sr,structures[track])+structure(pos[track]/sr,structures[tracks]))/2)
        # Increment the current position by the gap variable specified by the user
        pos[track] += gap_sam
    print "Added %d events" % _nevents
          
# ---------------
# plot_audio
# Just a debug function to plot a wave graph of the given audio
# ---------------  
def plot_audio(fig,aud,_sr,title,sec):
    plt.figure(fig)
    plt.title(title)
    if sec:
        # Calculate the x axis values if it's going to be in seconds
        time = np.linspace(0, len(aud)/_sr, num=len(aud))
        plt.plot(time,aud)
    else:
        plt.plot(aud)

# ---------------
# plot_lines
# Adds vertical lines to the current plot
# (used to display where the events are taken from)
# ---------------
def plot_lines(fig,lines,colours,lw=1):
    _colours = ['r','k','c','y','b','g','m']
    for i in range(0,len(lines)):
        plt.axvline(x=lines[i],linewidth=lw,color=_colours[colours[i]])

# ---------------
# select_events
# Selects a specified number of events randomly from the source audio
# and groups them with the K-Means clustering algorithm using
# a specified number of randomly selected spectral features (groups of spectrogram bins)
# ---------------
def select_events(nevents,nfeatures):
    global groups
    fftbins = 8192
    featurewidth = 16
    print "Selecting %d random spectral features.." % nfeatures
    feature_bins = np.random.randint(featurewidth/2,(fftbins/8),nfeatures)
    print "Selecting %d random audio events.." % nevents
    events = np.random.randint(0,len(faudio)-grain_mid,nevents)
    # Initialise features array with the first variable as index
    features = np.zeros((nfeatures+1,nevents))
    features[0] = np.arange(0,nevents)
    print "Computing audio event spectrograms.."
    # For each event..
    for i in range(0,nevents):
        # Calculate spectrogram for the event
        mags = abs(rfft(faudio[events[i]:min(events[i]+grain_mid,len(faudio))]*sig.hann(grain_mid),fftbins))
        mags = 20*log10(mags) # dB
        mags -= max(mags) # normalise to 0dB max
        # Calculate each feature for this event
        for j in range(0,nfeatures):
            features[j+1][i] = abs(np.mean(abs(mags[(feature_bins[j]-featurewidth/2):(feature_bins[j]+featurewidth/2)])))
    print "Clustering events with K-Means algorithm.."
    groups = kmeans(np.transpose(features),tracks,minit='points',iter=30)[1]
    return [events,groups]

# ---------------
# structure
# Calculate a point on the chance structure curve for the given track at the given time
# ---------------
def structure(time,str):
    return abs(np.sin(time*str[0]+str[2]) + np.sin(time*str[1]+str[3]))/2.

# ---------------
# ---------------
# Actual program starts here
# ---------------

# Select events from source audio and fill the group variables
[ev,g] = select_events(num_events,num_features)
event_groups = []
for i in range(0,tracks):
    event_groups.append([])
for i in range(0,num_events):
    event_groups[g[i]] = np.append(event_groups[g[i]],int(ev[i]))
    
# Generate structure curves
structures = []
for i in range(0,tracks+1):
    structures.append(np.random.uniform(-0.1,0.1,4))    

# Turn this on to view an example of a structure (controlling the overall amplitude)
plot_structure_example = False
if plot_structure_example:
    strex_x = np.arange(0,int(length_sec)*10)/10.
    strex_y = []
    for i in range(0,len(strex_x)):
        strex_y.append(structure(strex_x[i],structures[tracks]))
    plt.figure(1)
    plt.title('Overall Structure')
    plt.plot(strex_x,strex_y)
    plt.show()

# Fill each track
for i in range(0,tracks):
    print "Generating track %d (%d distinct audio events).." % ((i+1),len(event_groups[i]))
    filltrack(i)

# Turn this on to export each track (as track_x.wav)
export_individual_tracks = False
if export_individual_tracks:
    export_tracks(False);
data = 0.
data = mixdown()
print "Exporting to %s.." % outputfile
wav.write(outputfile,44100,data)
print "Done! Enjoy :)"

# Turn this on the plot the source audio with events from each track shown by lines
# (this will crash with more than 7 tracks because it will run out of colours)
plot_source_audio = False
if plot_source_audio:
    plot_audio(1,faudio,sr,'Source Audio',False)
    plot_lines(1,ev,g)
    plt.show()

# Turn this on the plot the generated audio file
plot_generated_audio = False
if plot_generated_audio:
    plot_audio(1,data,sr,'Generated Audio',True)
    plt.show()
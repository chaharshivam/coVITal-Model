import matplotlib.pyplot as plt
%matplotlib inline
import librosa
import librosa.display
import IPython.display as ipd
import os
import pandas as pd
import numpy as np

metadata = pd.read_csv('clinical\\labels.csv')
metadata.head(10)

# COVID Positive cough audio waveplot
filename_pos="clinical\original\pos\\pos-0421-084-cough-m-50.mp3"
plt.figure(figsize=(14,5))
data_pos,sample_rate_pos=librosa.load(filename_pos)
librosa.display.waveplot(data_pos,sr=sample_rate_pos)
ipd.Audio(filename_pos)

# Normalized Sample Rate
print(sample_rate_pos)

# Audio data converted to array
print(data_pos)

# COVID Negative cough audio waveplot
filename_neg="clinical\\original\\neg\\neg-0421-088-cough-f-66.mp3"
plt.figure(figsize=(14,5))
data_neg,sample_rate_neg=librosa.load(filename_neg)
librosa.display.waveplot(data_neg,sr=sample_rate_neg)
ipd.Audio(filename_neg)

# Normalized Sample Rate
print(sample_rate_pos)

# Audio data converted to array
print(data_pos)

# Mel spectogram of COVID positive cough audio
y, sr = librosa.load(filename_pos)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
print(ps.shape)

librosa.display.specshow(ps,y_axis='mel',x_axis='time')

## Mel spectogram of COVID Negative cough audio
y, sr = librosa.load(filename_neg)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
print(ps.shape)

librosa.display.specshow(ps,y_axis='mel',x_axis='time')


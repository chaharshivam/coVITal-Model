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


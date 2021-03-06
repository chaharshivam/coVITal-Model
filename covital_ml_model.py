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

# COVID Negative cough audio spectrum
X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14,5))
librosa.display.specshow(Xdb,sr=sr,x_axis='time',y_axis='hz')

# COVID Positive cough audio spectrum
y, sr = librosa.load(filename_pos)
X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14,5))
librosa.display.specshow(Xdb,sr=sr,x_axis='time',y_axis='hz')

# Feature extraction using MFCC
# Here we will be using Mel-Frequency Cepstral Coefficients(MFCC) from the audio samples. The MFCC summarises the frequency distribution across the window size, so it is possible to analyse both the frequency and time characteristics of the sound. These audio representations will allow us to identify features for classification.

# Extracting features of one file (sample)
filename_pos="clinical\original\pos\\pos-0421-084-cough-m-50.mp3"
data_pos,sample_rate_pos=librosa.load(filename_pos)
mfccs = librosa.feature.mfcc(y=data_pos, sr=sample_rate_pos, n_mfcc=40)
print(mfccs.shape)
print(mfccs)

# Extracting features of all audio files
def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_best')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features

from tqdm import tqdm
### Now we iterate through every audio file and extract features
### using Mel-Frequency Cepstral Coefficients
audio_dataset_path='clinical\\original\\'
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),str(row["folder"])+'\\',str(row["cough_filename"]))
    final_class_labels=row["corona_test"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])


### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head(20)

### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

# Label Encoding

#y=np.array(pd.get_dummies(y))
# Label Encoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train)
print(y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Model Creation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

# No of classes
num_labels=y.shape[1]

model=Sequential()

###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

# Training model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# Model accuracy
test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

# Testing
print(X_train[1])
print(np.argmax(model.predict(X_train), axis=-1))

# Testing with Some audio recordings of me coughing
filename="clinical\\shivam_cough.mp3"
audio, sample_rate = librosa.load(filename, res_type='kaiser_best')
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)
predicted_label=model.predict_classes(mfccs_scaled_features)
print(predicted_label)
prediction_class = labelencoder.inverse_transform(predicted_label)
print(prediction_class)


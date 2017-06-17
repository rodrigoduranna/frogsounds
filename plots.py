import glob
import os
import librosa
import numpy as np
import time
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.pyplot import specgram


plt.style.use('ggplot')
#Tamanho dos plots . Caso as imagens estiverem muito grandes ou pequenas, alterar aqui
H_SIZE = 10
V_SIZE = 22
DDPI = 96
#Parametros de fonte dos graficos
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13
#carrega os arquivos de som 
def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds
#plota o grafico de waveform
def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(H_SIZE,V_SIZE), dpi = DDPI)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f),sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=12)
    plt.show()
#plota o grafico de espectograma    
def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(H_SIZE,V_SIZE), dpi = DDPI)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=12)
    plt.show()
#plota o grafico de power log espectograma
def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(H_SIZE,V_SIZE), dpi = DDPI)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=12)
    plt.show()
#arquivos de som a serem criados os graficos
sound_file_paths = ["sapo1.wav"]
sound_names = ["Sapo-folha (Rhinella scitula) - Track 1"]
#carrega os arquivos de som
raw_sounds = load_sound_files(sound_file_paths)
#plota os graficos

plot_waves(sound_names,raw_sounds)
plot_specgram(sound_names,raw_sounds)
plot_log_power_specgram(sound_names,raw_sounds)
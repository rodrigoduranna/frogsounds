import glob
import os
import librosa
import numpy as np
import time
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.pyplot import specgram

comeco = time.time()

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
#raw_sounds = load_sound_files(sound_file_paths)
#plota os graficos

#plot_waves(sound_names,raw_sounds)
#plot_specgram(sound_names,raw_sounds)
#plot_log_power_specgram(sound_names,raw_sounds)



print("Extraindo caracteristicas ...")

#extrai as caracteristicas de um arquivo de som
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name) #extrai o numero de amostras do arquivo
    stft = np.abs(librosa.stft(X)) #Short time Fourier transform do arquivo de som
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) #MFCSS do arquivo de som
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0) #chromatograma do arquivo da STFT
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0) #Mel espectograma do araquivo de som
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0) #Contraste espectral da STFT
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0) #Tonnetz do arquivo de som
    return mfccs,chroma,mel,contrast,tonnetz

#procura todos os arquviso com a extensao .wav dentro de uma pasta e extrai as caracteristicas de cada arquivo
def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print("Extraindo caracteristicas de ", fn)
            try:
                mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn) #pega as caracteristicas de um arquivo de som
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz]) #coloca todas em formato de pilha
                features = np.vstack([features,ext_features]) #serializa
                labels = np.append(labels, fn.split('/')[2].split('-')[1]) #cria os labels para o arquivo
            except: 
                print("Erro ao processar o arquivo", fn) #caso o arquivo esteja corrompido
                pass
    return np.array(features), np.array(labels, dtype = np.int)

#
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

#pasta onde estão os arquivos a serem categorizados
parent_dir = 'Sound-Data'

#sub_dirs = ['fold1','fold2','fold3']

#sub diretorios a serem pesquisados
sub_dirs = ['fold1']

print("Lendo arquivos de som ...")
features, labels = parse_audio_files(parent_dir,sub_dirs)

print("Codificando caracteristicas ...")

labels = one_hot_encode(labels)

train_test_split = np.random.rand(len(features)) < 0.70
train_x = features[train_test_split]
train_y = labels[train_test_split]
test_x = features[~train_test_split]
test_y = labels[~train_test_split]

print("Treinando a rede neural ...")


import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

training_epochs = 5000 #numero de iteracoes para o treinamento ...
n_dim = features.shape[1]
n_classes = 10
n_hidden_units_one = 280 
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

#inserindo valores padrao nas variaveis
X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)


W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)


W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.global_variables_initializer()

print("Calculando a funcao de custo ...")

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1])) 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None

print("Iniciando sessao do TensorFlow ...")
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):            
        print("Epoch", epoch)
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:train_x,Y:train_y})
        cost_history = np.append(cost_history,cost)
    
    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: test_x})
    y_true = sess.run(tf.argmax(test_y,1))

print(" Plotando a funcao de custo ...")
fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.ylabel("Custo")
plt.xlabel("Iteracoes")
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
print ("Precisao (F-Score):", round(f,3))

fim = time.time()
print("Tempo de processamento", fim-comeco)
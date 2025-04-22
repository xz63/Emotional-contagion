# Emotional-contagion
LSTM model predict  a face watcher's emotion from the emotion of a face
a person watch some funny or sad videos  another person is watch him
using LSTM to predict the face watcher's emotion from the movie watcher's face
to answer: weather the emotion is contagious? what is the dynamic of such contagion
The data set is matlab mat file 
data/AUAll2.mat  including  AUAll1 and AUAll2, the movie watcher and face watcher, respectively
data/info12.mat  including info12    the colums are:  pairs,   c 123456, b 1-6 block
c 123  AUAll1 are from subject A AUAll2 are from subject B  because in those experiment, A is watching movie B is watch A's face
c 456  AUAll1 are from subject B AUAll2 are from subject A  vise versa
each block 545 for 18 seconds, there are 546 data sets from 22 pairs

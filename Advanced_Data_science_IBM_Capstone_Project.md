
# Advanced Data Science IBM - Capstone Project 


## Title of the project: Automatic scoring of the Electroencephalography (EEG) data

#### What is a EEG signal?      

* Based on [wikipedia](https://en.wikipedia.org/wiki/Electroencephalography), Electroencephalography (EEG) is an electrophysiological monitoring method to record electrical activity of the brain. It is typically noninvasive, with the electrodes placed along the scalp, although invasive electrodes are sometimes used, as in electrocorticography. EEG measures voltage fluctuations resulting from ionic current within the neurons of the brain.  
   
  
### What is our data?      

* Our data is an EEG signals from 3 electrodes collated on the mouse brain. To cover general brain activity first electrod is located on oscipital cortex, second of on frontal and third one on temporal cortex.    
   
   
### What is the use case of EEG signals?      

* Again based on [wikipedia](https://en.wikipedia.org/wiki/Electroencephalography), EEG signals have wide applications in clinical studies and research. EEG is most often used to diagnose epilepsy, which causes abnormalities in EEG readings. It is also used to diagnose sleep disorders, depth of anesthesia, coma, encephalopathies, and brain death. EEG used to be a first-line method of diagnosis for tumors, stroke and other focal brain disorders.   
However in our case, we use EEG signals to estimate sleep stage. If you need more information about relation of eeg signal to sleep stages please check this short [video](https://www.youtube.com/watch?v=NO-iUU8PIcE).   
   
     

### Solution algorithms   
   
1. Machine Learning based algorithm,      
   1.1 Support Vector Machines (SVM)   
   1.2 Ensemble methods   
       1.2.1 AdaBoost   
       1.2.2 ...   
   1.3 Linear models   
   1.4 ...   

2. Deep Learning based algorithms,   
   2.1 1D convolutional NN
   2.2 LSTM 
   2.3 GRU
   
   
   
      
### Dataset assessment    
* EEG signals contain several frequencies   
![](https://openi.nlm.nih.gov/imgs/512/127/3995691/PMC3995691_1475-925X-13-28-1.png)
Taken from [link](https://openi.nlm.nih.gov/imgs/512/127/3995691/PMC3995691_1475-925X-13-28-1.png)   
however not all frequency range is usefull for us, then first we need to filter signals in specific range.   
![](https://www.mdpi.com/electronics/electronics-08-01387/article_deploy/html/images/electronics-08-01387-g005.png)
Taken from [link](https://www.mdpi.com/electronics/electronics-08-01387/article_deploy/html/images/electronics-08-01387-g005.png)   
   
* Then when we have cleaned signals we can check all signals explain similar varances or normalization is necessary.   
![](Filtered_Signals.png)   





```python
from sklearn.preprocessing import StandardScaler   
scaler = StandardScaler()   
scaled_data = scaler.fit_transorm(eeg_signals)   
```

* In the next step we need to check how is our labels distribution. do we have balanced or imbalanced data set.   
![](Labels.png)   
* It is clear that we have imbalanced data set and we need to correct data distribution. One possible solution for this is sampling technique.   
* Additionally we can remove artifact group which is not important for us.   
![](sampled.png)

### Algorithm Selection   
* To do analysis I use deep neural network based algorithm.   
* For first model I use 1D CNN   
* Below you see summary of the model (In this notebook I not discussing model itself)   


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_layer (InputLayer)     (None, 10000, 3)          0         
    _________________________________________________________________
    conv1 (Conv1D)               (None, 10000, 16)         304       
    _________________________________________________________________
    batch1 (BatchNormalization)  (None, 10000, 16)         64        
    _________________________________________________________________
    active1 (Activation)         (None, 10000, 16)         0         
    _________________________________________________________________
    conv2 (Conv1D)               (None, 10000, 16)         1552      
    _________________________________________________________________
    batch2 (BatchNormalization)  (None, 10000, 16)         64        
    _________________________________________________________________
    active2 (Activation)         (None, 10000, 16)         0         
    _________________________________________________________________
    max1 (MaxPooling1D)          (None, 2500, 16)          0         
    _________________________________________________________________
    drop1 (Dropout)              (None, 2500, 16)          0         
    _________________________________________________________________
    conv3 (Conv1D)               (None, 2500, 32)          2080      
    _________________________________________________________________
    batch3 (BatchNormalization)  (None, 2500, 32)          128       
    _________________________________________________________________
    active3 (Activation)         (None, 2500, 32)          0         
    _________________________________________________________________
    conv4 (Conv1D)               (None, 2500, 32)          4128      
    _________________________________________________________________
    batch4 (BatchNormalization)  (None, 2500, 32)          128       
    _________________________________________________________________
    active4 (Activation)         (None, 2500, 32)          0         
    _________________________________________________________________
    max2 (MaxPooling1D)          (None, 625, 32)           0         
    _________________________________________________________________
    drop2 (Dropout)              (None, 625, 32)           0         
    _________________________________________________________________
    conv5 (Conv1D)               (None, 625, 64)           8256      
    _________________________________________________________________
    batch5 (BatchNormalization)  (None, 625, 64)           256       
    _________________________________________________________________
    active5 (Activation)         (None, 625, 64)           0         
    _________________________________________________________________
    conv6 (Conv1D)               (None, 625, 64)           16448     
    _________________________________________________________________
    batch6 (BatchNormalization)  (None, 625, 64)           256       
    _________________________________________________________________
    active6 (Activation)         (None, 625, 64)           0         
    _________________________________________________________________
    max3 (MaxPooling1D)          (None, 313, 64)           0         
    _________________________________________________________________
    drop3 (Dropout)              (None, 313, 64)           0         
    _________________________________________________________________
    conv7 (Conv1D)               (None, 313, 64)           16448     
    _________________________________________________________________
    batch7 (BatchNormalization)  (None, 313, 64)           256       
    _________________________________________________________________
    active7 (Activation)         (None, 313, 64)           0         
    _________________________________________________________________
    conv8 (Conv1D)               (None, 313, 64)           16448     
    _________________________________________________________________
    batch8 (BatchNormalization)  (None, 313, 64)           256       
    _________________________________________________________________
    max4 (MaxPooling1D)          (None, 157, 64)           0         
    _________________________________________________________________
    drop4 (Dropout)              (None, 157, 64)           0         
    _________________________________________________________________
    conv9 (Conv1D)               (None, 157, 128)          32896     
    _________________________________________________________________
    batch9 (BatchNormalization)  (None, 157, 128)          512       
    _________________________________________________________________
    active9 (Activation)         (None, 157, 128)          0         
    _________________________________________________________________
    conv10 (Conv1D)              (None, 157, 128)          65664     
    _________________________________________________________________
    batch10 (BatchNormalization) (None, 157, 128)          512       
    _________________________________________________________________
    active10 (Activation)        (None, 157, 128)          0         
    _________________________________________________________________
    max5 (MaxPooling1D)          (None, 79, 128)           0         
    _________________________________________________________________
    drop5 (Dropout)              (None, 79, 128)           0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 79, 128)           74112     
    _________________________________________________________________
    first_bn (BatchNormalization (None, 79, 128)           512       
    _________________________________________________________________
    bidirectional_2 (Bidirection (None, 79, 32)            13920     
    _________________________________________________________________
    sec_bn (BatchNormalization)  (None, 79, 32)            128       
    _________________________________________________________________
    bidirectional_3 (Bidirection (None, 79, 16)            1968      
    _________________________________________________________________
    th_bn (BatchNormalization)   (None, 79, 16)            64        
    _________________________________________________________________
    first_f (Flatten)            (None, 1264)              0         
    _________________________________________________________________
    first_d (Dense)              (None, 4)                 5060      
    =================================================================
    Total params: 262,420
    Trainable params: 260,852
    Non-trainable params: 1,568
    _________________________________________________________________
    


```python
model.compile(optimizer='Adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])
model.fit(eeg_data, to_categorical(labels), validation_split=0.2, epochs= 260, batch_size=128)
```

* To train this model I used Adam optimizer with categorical cross entropy loss. I use loss and accuacy as metrics for my training.   
* I used 20% of my train data for validation and trained my network for 260 epochs with 128 batch size.   
* This is accuracy graph.   
![](accuracy.png 'accuracy')   
* Blue line shows training accuracy and Red line indicates validation set accuracy.

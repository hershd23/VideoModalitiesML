# VideoModalitiesML

A set of ML/DL pipelines and models used to analyse various modalities of video data.

These pipelines have been tested on/ used to analyse the First Impressions V2 dataset(CVPR' 17), http://chalearnlap.cvc.uab.es/dataset/24/description/

## Modalities

### Audio

#### Audio_Features_Model.ipynb
* Input :- Requires audio features (CSV file) extracted using Librosa or any other audio feature extracting library.
* Models :- Contains implementation of Linear Regression and Random Forest models on audio features (Sklearn).

#### Audio_Features_Model.ipynb
* Input :- Requires audio spectrograms using https://github.com/swharden/Spectrogram or any other such library.
* Models :- Used the VGG11 pretrained model (Pytorch) with appended linear layers, to give the score in the desired format.

### Text

#### Text_BOWRegression.ipynb
* Input :- Uses the transcript data directly. No other preprocessing requires apart from feeding in the correct features from the Pandas dataframe
* Models :- Uses the SVR and Random Forest Regressor models (Sklearn), also uses NLTK to process the text data in the notebook itself.

#### Text_LSTMRegression.ipynb
* Input :- Uses the transcript data directly. No other preprocessing requires apart from feeding in the correct features from the Pandas dataframe
* Models :- Uses a Single layer BiLSTM model. Dataloading, auxiliary preprocessing and vector embedding integration facilitated using torchtext.

### Visual

#### Preprocessing
Use OpenCV or similar libraries to generate relevant frames from the video beforehand (Example scripts will be released soon) . The will be the inputs to the following models

#### Video_2d_cnn.ipynb
* Input :- Only one representative frame will be the input here, generally used as baselines in video models.
* Models :- Uses a pretrained 2D CNN model with appended linear layers, to give the score in the desired format. 

#### Video_3d_cnn.ipynb
* Input :- A set of 16 frames, in chronological order serve as input here
* Models :- Uses a pretrained 3D CNN model (by Facebook) with appended linear layers, to give the score in the desired format.

#### Video_LRCN.ipynb
* Input :- A set of 40 frames, in chronological order serve as input here
* Models :- Uses a pretrained ResNet 50 encoder model followed by an LSTM decoder with appended linear layers, to give the score in the desired format.

## Output and Metrics
As these models/pipelines have been trained on the First Impressions V2 dataset, the output is in the form of IOCEAN traits (Interview score + Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism traits). These were given as a real value between 0-1.

The loss used in the Deep Learning models is generally L1 or L2 (MSE) loss. Since we have used pytorch, adapting the code to a different loss function should be as easy as changing the function call.

The metric used is 1-MAE (Mean absolute error), used here http://chalearnlap.cvc.uab.es/dataset/24/results/49/

## On issues
Feel free to post issues if you find a bug and/or to suggest changes to the pipelines or models.

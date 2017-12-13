# Music-Classification
Build music classification Neural Network models using spectrograms as input data with Python 3.6 + Tensorflow + Keras

# Requirements
## Dataset and utilery
In order to get the dataset used and some utilery, you'll have to clone the fma github [repo](https://github.com/mdeff/fma)  inside this project's directory. Download the dataset on the data section of the repo and place it inside the `fma` directory.

## Training a model
The model implemented is based on Sander Dielemann's [architecture](http://benanne.github.io/2014/08/05/spotify-cnns.html) and is implemented on `models.Dielemann.py`.
To train it, from your terminal, at the root level of this repo, run 

`python -m training.Dielmann_windows hop_length_param number_of_windows_param model_name_param number_of_epochs_param`, for example, `python -m training.Dielmann_windows 355 10 "hop_length_355.h5" 5`

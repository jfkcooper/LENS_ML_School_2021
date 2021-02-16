# Lecture 10: unsupervised and reinforcement learning

## Environment

Tutorials in this lecture are not very computationaly intense and can be run from any environment: your local computer, google colab or JSC cluster.

### Local machine

To be able to run this tutorial on your local machine, you will need to install the packages listed in the [requirements.txt](requirements.txt) file. One can do it with the following command:

```
pip install -r requirements.txt
```

or

```
pip install -r requirements.txt --user
```
if you want to install these packages locally, for a particular user.


### JSC cluster

To get the lecture materials to your home directory, run in terminal:

```
cp -a /p/project/training2101/LENS_ML_School_2021 $HOME
```

You do not to setup anything special to be able to run the first part of the tutorial. However, for the *Reinforcement learning* part additional modules need to be loaded. The easiest way to do that is to use the script [start_jupyter-jsc.sh](start_jupyter-jsc.sh). It should be put into your `$HOME/.jupyter/` directory. To do so, run in the terminal: 

```
cp LENS_ML_School_2021/lecture_10/start_jupyter-jsc.sh $HOME/.jupyter/

```

After that you will need to **restart your jupyter lab**: go to the menu `File->Hub Control Panel` and press first `Stop` and then `Start` buttons for your Jupyter lab.


## Disclaimer

This lecture is just a brief introduction and does not pretend to be a comprehensive guide to the topic. The topic is too huge and it is impossible to cover it in a given time. Thus, the aim is to present you some methods and let you "feel the taste" of this amazing science :)

I reuse a lot of code and images produced by other authors. There is always should be link to the source or original. If I have overseen something, please let me know, I will insert the link. Most of third-party images are not distributed with this tutorial: they are just linked and are downloaded from the source. If the source will change, the image can disappear. If you want to keep it for your personal use, please download the image from the source page.
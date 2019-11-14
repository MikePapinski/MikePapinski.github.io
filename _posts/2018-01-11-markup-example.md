---
layout: post
title:  "The easiest way to use Keras on GPU with Docker."
author: MikePapinski
categories: [Docker, Deep Learning, Machine Learning, Python]
image: assets/images/posts/1/post_1.jpg
---


# Introduction

If you are reading this, you are probably struggling with running your super Keras deep learning models on your GPU. But guess what, I was at the same place a few months ago an I couldn’t find any good tutorial on how to properly set up your Keras deep learning GPU environment. It took me some time, trial & error before I figure out how to utilize my GPU while training models, but I will save you this effort. If you stay with me until the end of this tutorial. You will have a fully working Keras environment on your computer up and running in 1-2 hours. Most of that time you will be waiting while your computer will be processing tasks !!!

<br>
![Avatar]({{ site.baseurl }}/assets/images/posts/1/1.jpg)
<br>


### Why you want your Keras to use GPU?
First of all, we should ask ourselves a question. Why do we need our Keras to use GPU? The answer is simple: The GPU has more computing power than a CPU. This is crucial when we are building complexed models and train them on large datasets. The most popular example is the Convolutional 2D model to classify images. On CPU training of image, the classifier would take ages but GPU can handle it quicker and more efficient. That is why we need Keras to communicate with GPU.
<br>

# What do we need?
![Req]({{ site.baseurl }}/assets/images/posts/1/2.jpg)

### Operating system :
Choose one from the below.
-Linux (Any type will work)
-Mac OS (Most updated version)
-Windows Professional (Please note that Windows Home edition will not work)

<br>

I prefer to work on Linux (Ubuntu or Xubuntu). For me, it is much easier to set up and control the Development environment but… the choice is yours.

### Nvidia driver installed :
I believe in Windows and Mac OS you can install it from .exe file downloaded from NVIDIA site [LINK](https://www.nvidia.com/Download/index.aspx).
But if you have Linux and you are lazy like me, you can just install it by using the below commands on the terminal.
```bash
sudo apt install nvidia-driver-390
```
And reboot the system.
```bash
$ sudo reboot
```
<br>

### Docker installed :
If you do not have docker installed on your machine, please stop now.
1. Go to this [LINK](https://docs.docker.com/install).
2. Follow the official Docker installation tutorial
3. Check if Docker works properly on your machine
4. Go back and follow this tutorial

<br>

### Docker image of KERAS GPU Environment.
This is a link to Github repository with the most up to date image I use personally to my projects.

<a class="social__link" target="_blank" rel="noopener noreferrer" href="https://github.com/MikePapinski/Keras_Jupyter_GPU_Docker">
     <svg viewBox="0 0 16 16" width="20px" height="20px"><path d="M7.999,0.431c-4.285,0-7.76,3.474-7.76,7.761 c0,3.428,2.223,6.337,5.307,7.363c0.388,0.071,0.53-0.168,0.53-0.374c0-0.184-0.007-0.672-0.01-1.32 c-2.159,0.469-2.614-1.04-2.614-1.04c-0.353-0.896-0.862-1.135-0.862-1.135c-0.705-0.481,0.053-0.472,0.053-0.472 c0.779,0.055,1.189,0.8,1.189,0.8c0.692,1.186,1.816,0.843,2.258,0.645c0.071-0.502,0.271-0.843,0.493-1.037 C4.86,11.425,3.049,10.76,3.049,7.786c0-0.847,0.302-1.54,0.799-2.082C3.768,5.507,3.501,4.718,3.924,3.65 c0,0,0.652-0.209,2.134,0.796C6.677,4.273,7.34,4.187,8,4.184c0.659,0.003,1.323,0.089,1.943,0.261 c1.482-1.004,2.132-0.796,2.132-0.796c0.423,1.068,0.157,1.857,0.077,2.054c0.497,0.542,0.798,1.235,0.798,2.082 c0,2.981-1.814,3.637-3.543,3.829c0.279,0.24,0.527,0.713,0.527,1.437c0,1.037-0.01,1.874-0.01,2.129 c0,0.208,0.14,0.449,0.534,0.373c3.081-1.028,5.302-3.935,5.302-7.362C15.76,3.906,12.285,0.431,7.999,0.431z"/></svg>
        Link to Github repository.
</a>

<br>

# Let's make it work.
![Start]({{ site.baseurl }}/assets/images/posts/1/3.jpg)



### Download a Docker image.
Ok, first we need to create a folder on your desktop called ‘DockerKeras’ and download a docker image from the Github repository. We will do it using git but you can also manually download it from Github, create a folder on the desktop and extract repo to this folder.

```bash
cd desktop
mkdir DockerKeras
cd DockerKeras
git clone https://github.com/MikePapinski/Keras_Jupyter_GPU_Docker
ls
```
 You should see this folder structure:
```
Keras_Jupyter_GPU_Docker
│   README.md
│   LICENSE
└───docker
│   │   Dockerfile
│   │   Makefile
│   │   README.md
│   │   theanorc
```

### Update Keras_Jupyter_GPU_Docker/docker/Makefile.
Go to Desktop -> Keras_Jupyter_GPU_Docker -> docker -> and edit file ‘Makefile’ The only thing you can change but do not have to is the DATA parameter

```bash
DATA?="${HOME}/data"
```

#### you got 2 options:
1. Change “${HOME}/data” into the path you want jupyter notebook to read as root.
2. Go to your Home directory and create inside a folder called “data”. jupyter will store all the files in there.

### Build the containers
Now we are all set to build our containers. Go inside the docker folder and trigger the method to build the containers. Please be aware that if you are building the container for the first time, it can take up to one hour for docker to complete the build.

```bash
cd Keras_Jupyter_GPU_Docker
cd docker
make notebook GPU=0 # or [ipython, bash]
```
Docker will start downloading all the required packages and you should see something like this on your console.
<br>
![Start]({{ site.baseurl }}/assets/images/posts/1/5.jpg)
<br>

When docker will finish building the container, you should this:
<br>
![Start]({{ site.baseurl }}/assets/images/posts/1/4.jpg)
<br>


<br>

### We are almost there…
OK, so now you got your docker container up and running! Just copy the link to your localhost and port 8888 that you will see on the very bottom of the last message on your console. In my case it will be:

```bash
http://127.0.0.1:8888/?token=9fd39b2da74d5da694157e03a54f1ca5cb04809d4dc134cd
```
And paste it to your browser. If you did all the previous steps correctly you should see this page.
<br>
![Start]({{ site.baseurl }}/assets/images/posts/1/6.jpg)
<br>
If you see the same screen it means that…
# You did it!
<br>
![Start]({{ site.baseurl }}/assets/images/posts/1/7.jpg)
<br>
<br>

### OK, you got your KERAS environment fully compatible with your GPU so what now?
# Why not test it out right away?


<br>
## To do that, we will use the most common model for beginners which is already implemented in KERAS library. FASHION MNIST.
<br>
![Start]({{ site.baseurl }}/assets/images/posts/1/8.jpg)
<br>

In the Jupyter editor, click on “New” and then on “Python 3”. You will open the new project window.
<br>
![Start]({{ site.baseurl }}/assets/images/posts/1/9.jpg)
<br>

Now copy and paste the below code to the first cell-like on the picture below.

```Python
import keras, tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

np.random.seed(1)
tf.set_random_seed(1)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

type(train_images)
train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig = plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i]])
fig
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(3, activation=tf.nn.relu),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=4)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```
<br>
![Start]({{ site.baseurl }}/assets/images/posts/1/10.jpg)
<br>

And click on “Run”. After that, our Fashion MNIST model will start to train. Please note that preparing GPU for the session usually takes some time for the first time, more or less than 5 min. After we the GPU will be prepared for the tensor training, It will launch in a few seconds every next time you train the model.

<br>
![Start]({{ site.baseurl }}/assets/images/posts/1/13.jpg)
<br>

We can now see that the model was trained successfully to 0.6947% accuracy. There are much better models for this dataset if you want to look, but our model was only for demonstration purposes.

# Few tips for Linux users!
<br>
### How to check if the training process goes on GPU?
if during the training of the model you will type to your console:
```bash
nvidia-smi
```
You will see the status of your GPU. If under Processes, you will see the process with type “C” and “/opt/conda/bin/python “, it means that you the Keras model training process is using your GPU.
<br>
![Start]({{ site.baseurl }}/assets/images/posts/1/11.jpg)
<br>
<br>

### How to check my CPU consumption during training/data loading?
Type to you console:
```bash
htop
```
You will see the current status of all your CPU usage.
<br>
![Start]({{ site.baseurl }}/assets/images/posts/1/12.jpg)
<br>
<br>

# This is the very end of this tutorial.
I wish you great ideas and great Deep Learning projects with your new build Keras GPU environment.
<br>
![Start]({{ site.baseurl }}/assets/images/posts/1/14.jpg)
<br>

### ps. If you want to quickly open the jupyter notebook.
Just open the folder with file “makefile” and run this command inside.
```bash
make notebook GPU=0 # or [ipython, bash]
```

<br>
<br>
### Like my stuff?
### Leave a comment.
### Get in touch.



<br>

<a class="social__link" target="_blank" rel="noopener noreferrer" href="https://github.com/{{ site.github_username }}">
  -
         <svg version="1.1" id="Capa_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 width="20px" height="20px" viewBox="0 0 60.734 60.733" style="enable-background:new 0 0 60.734 60.733;"
	 xml:space="preserve">
	<path d="M57.378,0.001H3.352C1.502,0.001,0,1.5,0,3.353v54.026c0,1.853,1.502,3.354,3.352,3.354h29.086V37.214h-7.914v-9.167h7.914
		v-6.76c0-7.843,4.789-12.116,11.787-12.116c3.355,0,6.232,0.251,7.071,0.36v8.198l-4.854,0.002c-3.805,0-4.539,1.809-4.539,4.462
		v5.851h9.078l-1.187,9.166h-7.892v23.52h15.475c1.852,0,3.355-1.503,3.355-3.351V3.351C60.731,1.5,59.23,0.001,57.378,0.001z"/>
        </svg>
            Facebook - Private
</a>

<a class="social__link" target="_blank" rel="noopener noreferrer" href="https://github.com/{{ site.github_username }}">
  -
         <svg version="1.1" id="Capa_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 width="20px" height="20px" viewBox="0 0 60.734 60.733" style="enable-background:new 0 0 60.734 60.733;"
	 xml:space="preserve">
	<path d="M57.378,0.001H3.352C1.502,0.001,0,1.5,0,3.353v54.026c0,1.853,1.502,3.354,3.352,3.354h29.086V37.214h-7.914v-9.167h7.914
		v-6.76c0-7.843,4.789-12.116,11.787-12.116c3.355,0,6.232,0.251,7.071,0.36v8.198l-4.854,0.002c-3.805,0-4.539,1.809-4.539,4.462
		v5.851h9.078l-1.187,9.166h-7.892v23.52h15.475c1.852,0,3.355-1.503,3.355-3.351V3.351C60.731,1.5,59.23,0.001,57.378,0.001z"/>
        </svg>
            Facebook - DivergenceFX fanpage
</a>



<a class="social__link" target="_blank" rel="noopener noreferrer" href="https://github.com/{{ site.github_username }}">
  -
     <svg height="20px" version="1.1" viewBox="0 0 20 16" width="20px" xmlns="http://www.w3.org/2000/svg" xmlns:sketch="http://www.bohemiancoding.com/sketch/ns" xmlns:xlink="http://www.w3.org/1999/xlink"><title/><desc/><defs/><g fill="none" fill-rule="evenodd" id="Page-1" stroke="none" stroke-width="1"><g fill="#000000" id="Icons-AV" transform="translate(-42.000000, -171.000000)"><g id="video-youtube" transform="translate(42.000000, 171.000000)"><path d="M18,0.4 C17.4,0.2 13.7,0 10,0 C6.3,0 2.6,0.2 2,0.4 C0.4,0.9 0,4.4 0,8 C0,11.6 0.4,15.1 2,15.6 C2.6,15.8 6.3,16 10,16 C13.7,16 17.4,15.8 18,15.6 C19.6,15.1 20,11.6 20,8 C20,4.4 19.6,0.9 18,0.4 L18,0.4 Z M8,12.5 L8,3.5 L14,8 L8,12.5 L8,12.5 Z" id="Shape"/></g></g></g></svg>
        Youtube - DivergenceFX
</a>



<a class="social__link" target="_blank" rel="noopener noreferrer" href="https://github.com/{{ site.github_username }}">
  -
     <svg viewBox="0 0 16 16" width="20px" height="20px"><path d="M8 0C5.827 0 5.555.01 4.702.048 3.85.088 3.27.222 2.76.42a3.908 3.908 0 0 0-1.417.923c-.445.444-.72.89-.923 1.417-.198.51-.333 1.09-.372 1.942C.008 5.555 0 5.827 0 8s.01 2.445.048 3.298c.04.852.174 1.433.372 1.942.204.526.478.973.923 1.417.444.445.89.72 1.417.923.51.198 1.09.333 1.942.372.853.04 1.125.048 3.298.048s2.445-.01 3.298-.048c.852-.04 1.433-.174 1.942-.372a3.908 3.908 0 0 0 1.417-.923c.445-.444.72-.89.923-1.417.198-.51.333-1.09.372-1.942.04-.853.048-1.125.048-3.298s-.01-2.445-.048-3.298c-.04-.852-.174-1.433-.372-1.942a3.908 3.908 0 0 0-.923-1.417A3.886 3.886 0 0 0 13.24.42c-.51-.198-1.09-.333-1.942-.372C10.445.008 10.173 0 8 0zm0 1.44c2.136 0 2.39.01 3.233.048.78.036 1.203.166 1.485.276.374.145.64.318.92.598.28.28.453.546.598.92.11.282.24.705.276 1.485.038.844.047 1.097.047 3.233s-.01 2.39-.05 3.233c-.04.78-.17 1.203-.28 1.485-.15.374-.32.64-.6.92-.28.28-.55.453-.92.598-.28.11-.71.24-1.49.276-.85.038-1.1.047-3.24.047s-2.39-.01-3.24-.05c-.78-.04-1.21-.17-1.49-.28a2.49 2.49 0 0 1-.92-.6c-.28-.28-.46-.55-.6-.92-.11-.28-.24-.71-.28-1.49-.03-.84-.04-1.1-.04-3.23s.01-2.39.04-3.24c.04-.78.17-1.21.28-1.49.14-.38.32-.64.6-.92.28-.28.54-.46.92-.6.28-.11.7-.24 1.48-.28.85-.03 1.1-.04 3.24-.04zm0 2.452a4.108 4.108 0 1 0 0 8.215 4.108 4.108 0 0 0 0-8.215zm0 6.775a2.667 2.667 0 1 1 0-5.334 2.667 2.667 0 0 1 0 5.334zm5.23-6.937a.96.96 0 1 1-1.92 0 .96.96 0 0 1 1.92 0z"></path></svg>
        Instagram - Private
</a>




<a class="social__link" target="_blank" rel="noopener noreferrer" href="https://github.com/{{ site.github_username }}">
  -
         <svg version="1.1" id="Capa_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 width="20px" height="20px" viewBox="0 0 510 510" style="enable-background:new 0 0 510 510;" xml:space="preserve">
                <path d="M459,0H51C22.95,0,0,22.95,0,51v408c0,28.05,22.95,51,51,51h408c28.05,0,51-22.95,51-51V51C510,22.95,487.05,0,459,0z
                    M153,433.5H76.5V204H153V433.5z M114.75,160.65c-25.5,0-45.9-20.4-45.9-45.9s20.4-45.9,45.9-45.9s45.9,20.4,45.9,45.9
                    S140.25,160.65,114.75,160.65z M433.5,433.5H357V298.35c0-20.399-17.85-38.25-38.25-38.25s-38.25,17.851-38.25,38.25V433.5H204
                    V204h76.5v30.6c12.75-20.4,40.8-35.7,63.75-35.7c48.45,0,89.25,40.8,89.25,89.25V433.5z"/>
            </svg>
            Linkedin - Resume
</a>



<a class="social__link" target="_blank" rel="noopener noreferrer" href="https://github.com/{{ site.github_username }}">
  -
     <svg height="20px" style="enable-background:new 0 0 512 512;" version="1.1" viewBox="0 0 512 512" width="20px" xml:space="preserve" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g id="comp_x5F_129-fiverr"><g><g><path d="M413.696,399.752V169.75H184.32v-14.375c0-23.779,19.295-43.124,43.004-43.124h43.008V26h-43.008     C156.191,26,98.306,84.044,98.306,155.375v14.375H40.962V256h57.342v143.752H40.962V486h200.705v-86.248H184.32V256h144.157     v143.752h-58.145V486h200.705v-86.248H413.696z"/><path d="M370.685,112.25c23.751,0,43.012-19.305,43.012-43.125c0-23.821-19.261-43.125-43.012-43.124     c-23.749,0-43.005,19.303-43.005,43.124C327.68,92.945,346.936,112.25,370.685,112.25z"/></g></g></g><g id="Layer_1"/></svg>
        Fiverr - Services
</a>




<a class="social__link" target="_blank" rel="noopener noreferrer" href="https://github.com/{{ site.github_username }}">
  -
     <svg enable-background="new 0 0 100 100" height="20px" id="Layer_1" version="1.1" viewBox="0 0 100 100" width="20px" xml:space="preserve" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><g><defs><rect height="100" id="SVGID_1_" width="100"/></defs><path d="M95,49.247c0,24.213-19.779,43.841-44.182,43.841c-7.747,0-15.025-1.98-21.357-5.455L5,95.406   l7.975-23.522c-4.023-6.606-6.34-14.354-6.34-22.637c0-24.213,19.781-43.841,44.184-43.841C75.223,5.406,95,25.034,95,49.247    M50.818,12.388c-20.484,0-37.146,16.535-37.146,36.859c0,8.066,2.629,15.535,7.076,21.611l-4.641,13.688l14.275-4.537   c5.865,3.851,12.891,6.097,20.437,6.097c20.481,0,37.146-16.533,37.146-36.858C87.964,28.924,71.301,12.388,50.818,12.388    M73.129,59.344c-0.273-0.447-0.994-0.717-2.076-1.254c-1.084-0.537-6.41-3.138-7.4-3.494c-0.993-0.359-1.717-0.539-2.438,0.536   c-0.721,1.076-2.797,3.495-3.43,4.212c-0.632,0.719-1.263,0.809-2.347,0.271c-1.082-0.537-4.571-1.673-8.708-5.334   c-3.219-2.847-5.393-6.364-6.025-7.44c-0.631-1.075-0.066-1.656,0.475-2.191c0.488-0.482,1.084-1.255,1.625-1.882   c0.543-0.628,0.723-1.075,1.082-1.793c0.363-0.717,0.182-1.344-0.09-1.883c-0.27-0.537-2.438-5.825-3.34-7.976   c-0.902-2.151-1.803-1.793-2.436-1.793c-0.631,0-1.354-0.09-2.076-0.09s-1.896,0.269-2.889,1.344   c-0.992,1.076-3.789,3.676-3.789,8.963c0,5.288,3.879,10.397,4.422,11.114c0.541,0.716,7.49,11.92,18.5,16.223   C63.2,71.177,63.2,69.742,65.186,69.562c1.984-0.179,6.406-2.599,7.312-5.107C73.398,61.943,73.398,59.792,73.129,59.344"/></g></svg>
        Whatsapp - chat
</a>


<a class="social__link" target="_blank" rel="noopener noreferrer" href="https://github.com/{{ site.github_username }}">
  -
     <svg viewBox="0 0 16 16" width="20px" height="20px"><path d="M7.999,0.431c-4.285,0-7.76,3.474-7.76,7.761 c0,3.428,2.223,6.337,5.307,7.363c0.388,0.071,0.53-0.168,0.53-0.374c0-0.184-0.007-0.672-0.01-1.32 c-2.159,0.469-2.614-1.04-2.614-1.04c-0.353-0.896-0.862-1.135-0.862-1.135c-0.705-0.481,0.053-0.472,0.053-0.472 c0.779,0.055,1.189,0.8,1.189,0.8c0.692,1.186,1.816,0.843,2.258,0.645c0.071-0.502,0.271-0.843,0.493-1.037 C4.86,11.425,3.049,10.76,3.049,7.786c0-0.847,0.302-1.54,0.799-2.082C3.768,5.507,3.501,4.718,3.924,3.65 c0,0,0.652-0.209,2.134,0.796C6.677,4.273,7.34,4.187,8,4.184c0.659,0.003,1.323,0.089,1.943,0.261 c1.482-1.004,2.132-0.796,2.132-0.796c0.423,1.068,0.157,1.857,0.077,2.054c0.497,0.542,0.798,1.235,0.798,2.082 c0,2.981-1.814,3.637-3.543,3.829c0.279,0.24,0.527,0.713,0.527,1.437c0,1.037-0.01,1.874-0.01,2.129 c0,0.208,0.14,0.449,0.534,0.373c3.081-1.028,5.302-3.935,5.302-7.362C15.76,3.906,12.285,0.431,7.999,0.431z"/></svg>
        Github - Open source projects
</a>

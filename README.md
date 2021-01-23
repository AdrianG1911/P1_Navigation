# P1_Navigation

## Project Details

In this environment the agent moves around a square space and gets +1 reward for collecting yellow bananas and -1 reward for collecting blue bananas, so the goal of the agent is to collect as many yellow bananas and avoid blue bananas in a limited amount of time.

The state space has 37 dimensions including the agents velocity and a ray based perception of objects in the agent's vision. The action space is descrete and has 4 possible actions, moving forward, backwards, turning left and turning right. The task is episodic and in order to solve the environment the agent must get and average score of at least 13 over 100 consecutive episodes.


## Downloading Dependencies

**Jupyter Notebook** is needed to open the Navigation.ipynb file. To get Jupyter Notebook download [Anaconda](https://www.anaconda.com/products/individual). Install it then open the command line and input `conda install -c conda-forge notebook` after installing Jupyter Notebook can be opened by inputing `jupyter notebook` in the command line. 

**Pytorch** can be installed by going to their [site](https://pytorch.org/), choosing the stable version and the platform you are using then typing the given command in the command line.

**Unity mlagents** can be installed by `inputing pip3 install torch==1.7.0 -f https://download.pytorch.org/whl/torch_stable.html` in the command line

**The Environment**
Download the environment from one of the links below.  You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

**Running the code**
To run the code put all the files in the same folder and open Navigation.ipynb with jupyter notebook. 
`` `` 

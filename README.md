# ME5406 Project 2


A reinforcement learning task for simultaneous lane-following and obstacle avoidance while handling interactions with other vehicles. The [highway_env API](https://github.com/eleurent/highway-env) is used for simulation.

<br/>

## Files

* [main.py](main.py) - Main python file for running training & testing sequences. Can be run with various options.

* [racetrack_env.py](racetrack_env.py) - Looped Race Track Environment for the RL problem.

* [agent/models.py](/agent/models.py) - Contains NN architectures that can be imported for various agents.

* [agent/DQN.py](/agent/DQN.py) - DQN Agent & its Variants

* [agent/PPO.py](/agent/PPO.py) - PPO Agent with Clipped Surrogate & GAE

* [requirements.txt](requirements.txt) - Conda environment for running the project.

&nbsp;

<i><b>NOTE:</b> A [similar version of racetrack_env.py](https://github.com/eleurent/highway-env/blob/master/highway_env/envs/racetrack_env.py) can be found in the original highway_env repo, and was contributed by us through https://github.com/eleurent/highway-env/issues/231.</i>

<br/>

## Installation  

First clone the repository 

```
git clone https://github.com/supperted825/ME5406P2.git
```

Then, create the conda environment with the required dependencies.

```
conda create --name <env> --file requirements.txt
conda activate <env>
```

To run experiments, run main.py while specifying options. See main file for available options.

```
python main.py [--opts]
```

<br/>

## Developing

Create a branch with the name of the feature that you are working on in ~/highway_env.

```
git co -b <branch name>
```

Now, you can edit the files as normal and develop.

To register changes, be sure to stage the files by running the following at the root of the repo folder.

```
git add .
```

Throughout, you can commit changes with the following command. This doesn't push your changes online yet.

```
git commit -m "your message"
```

To submit your changes to the main repo, you will have to push with:

```
git push
```

Now, if you head back to the repo online, you should see "Compare and Pull Request". Click on it and enter a short description of your changes. Then you can select a reviewer and submit the code for merging with a pull request.
# ME5406 Project 2 - RL Agents for RaceTrackEnv

A reinforcement learning project for simultaneous lane-following and obstacle avoidance while handling interactions with other vehicles. The [highway_env API](https://github.com/eleurent/highway-env) is used for simulation.

We train the agents to perform two types of tasks. First, to simply learn lane following and traverse the track. Next, we introduce three randomly spawned non-agent vehicles around the train that move at a slower speed. The agent must simultaneously learn to overtake and traverse the track.

<br>

# Demo

We successfully train DQN & PPO on both tasks as shown below:

<p align="center">
    <img src="./media/DQN2.gif" width="70%", height="70%"<br/>
    <em><br>The DQN Agent Performing Laning & Overtaking.</em>
</p>



# Files

* [main.py](main.py) - Main python file for running training & testing sequences. Can be run with various options.

* [racetrack_env.py](racetrack_env.py) - Looped Race Track Environment for the RL problem.

* [agent/models.py](/agent/models.py) - Contains NN architectures that can be imported for various agents.

* [agent/DQN.py](/agent/DQN.py) - DQN Agent & its Variants

* [agent/PPO.py](/agent/PPO.py) - PPO Agent with Clipped Surrogate & GAE

* [requirements.txt](requirements.txt) - Conda environment for running the project.

* [models](/models/) - Trained Agent Models (Keras Model API) that can be loaded for demo.

&nbsp;

<i><b>NOTE:</b> A [similar version of racetrack_env.py](https://github.com/eleurent/highway-env/blob/master/highway_env/envs/racetrack_env.py) can be found in the original highway_env repo, and was contributed by us through https://github.com/eleurent/highway-env/issues/231. Our env uses a slightly different reward structure than the original to facilitate training.</i>

<br/>

# Load & Run Models

First, please ensure you have the correct requirements installed.

```
conda create --name <env> --file requirements.txt
conda activate <env>
```

To load and run each model sequentially with visualisation, you may use the following commands:

```
python3 main.py --mode test --agent DQN --load_model ./models/DQN1.model --save_video
python3 main.py --mode test --agent DQN --load_model ./models/DQN2.model --spawn_vehicles 3 --save_video
python3 main.py --mode test --agent PPO --load_model ./models/PPO1.model --save_video
python3 main.py --mode test --agent PPO --load_model ./models/PPO2.model --spawn_vehicles 3 --save_video
```

# Training

To run your own experiments, please look at main.py for the available options. For example, we train DQN on Task 2 with the following command:

```
python3 ./main.py --agent DQN \
                --exp_id dqn2 \
                --num_episodes 5000 --batch_size 256 \
                --epsilon 0.6 --min_epsilon 0 \
                --lr 0.00005 --lr_decay \
                --arch Identity --fc_layers 3 \
                --spawn_vehicles 3
```
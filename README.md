# RL Agents for RaceTrackEnv

A reinforcement learning project for simultaneous lane-following and obstacle avoidance while handling interactions with other vehicles. The [highway_env API](https://github.com/eleurent/highway-env) is used for simulation.

We train the agents to perform two types of tasks. In task 1, to simply learn lane following and traverse the track. For Task 2, we introduce three randomly spawned non-agent vehicles around the track that move at a slower speed. The agent must now simultaneously learn to overtake and traverse the track.

Available agents include Double DQN, PPO, A3C, DDPG. Special thanks to project-mates [@hwchua0209](@hwchua0209) and [@jeremyxychew](@jeremyxychew) for their work on the A3C & DDPG agents respectively.

<br>

<p align="center">
    <img src="./media/PPO2.gif" width="80%", height="80%"<br/>
    <em><br>The PPO Agent Performing Laning & Overtaking.</em>
</p>



# Files

We use Python 3.8 with Tensorflow v2.6.0 and HighwayEnv v1.4. Additional core dependencies include Tensorflow Probability v0.14.1 and tqdm v4.62.3.

<br>

* [main.py](main.py) - Main python file for running training & testing sequences. Can be run with various options.

* [racetrack_env.py](racetrack_env.py) - Looped Race Track Environment for the RL problem.

* [agent/models.py](/agent/models.py) - Contains NN architectures that can be imported for various agents.

* [agent/DQN.py](/agent/DQN.py) - Deep Q-Network (DQN) Agent & its Variants

* [agent/PPO.py](/agent/PPO.py) - Proximal Policy Optimisation (PPO) Agent with Clipped Surrogate & GAE

* [agent/A3C.py](/agent/A3C.py) - Asynchronous Advantage Actor-Critic (A3C) Agent

* [agent/DDPG.py](/agent/DDPG.py) - Deep Deterministic Policy Gradient (DDPG) Agent

* [requirements.txt](requirements.txt) - Pip-compatible requirements for running the project.

* [models](/models/) - Trained Agent Models (Keras Model API) that can be loaded for demo.

&nbsp;

<i><b>NOTE:</b> A [similar version of racetrack_env.py](https://github.com/eleurent/highway-env/blob/master/highway_env/envs/racetrack_env.py) can be found in the original highway_env repo, and was contributed by us through https://github.com/eleurent/highway-env/issues/231. Our env uses a slightly different reward structure than the original to facilitate training.</i>

<br>

# Load & Run Models

First, please ensure you have the correct requirements installed.

```
conda create --name <env> python=3.8
conda activate <env>
pip install -r requirements.txt
```

To load and run each available model sequentially with visualisation, use the following commands.

```
python3 main.py --mode test --agent DQN --load_model ./models/DQN1.model --save_video
python3 main.py --mode test --agent DQN --load_model ./models/DQN2.model --spawn_vehicles 3 --save_video

python3 main.py --mode test --agent PPO --load_model ./models/PPO1.model --save_video
python3 main.py --mode test --agent PPO --load_model ./models/PPO2.model --spawn_vehicles 3 --save_video

python3 main.py --mode test --agent A3C --load_model ./models/A3C1.model --save_video

python3 main.py --mode test --agent DDPG --load_model ./models/DDPG1.model --save_video
python3 main.py --mode test --agent DDPG --load_model ./models/DDPG2.model --spawn_videos 3 --save_video
```

The above experiments with vehicles spawn one fixed vehicle and two random vehicles. If you want to try more vehicles, please adjust the --spawn_vehicles parameter. For all random vehicles, pass the --all_random flag.

In some cases, the trained agent may fail in the presence of other vehicles. Admittedly, these policies are not entirely robust and can be improved with further training. If this happens, please restart the run :-).

<br>

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

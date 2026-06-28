# AI HAnds-on 3rd assignemnt - Reinforcement learning

## Task 1: Tabular Q-Learning Baseline

Q-Table Size: To apply tabular Q-learning to the continuous CartPole-v1 environment, the 4-dimensional state space was discretized into 6 uniform bins per variable. This results in $6^4 = 1296$ unique discrete states. Because there are 2 possible actions, the **final Q-table size** is $1296 \times 2 = 2592$ state-action pairs.

### Results

The following image show the graph of

<img src="results/task1_tabular_qlearning.png"
     alt="DQN ablation study graph"
     width="700" />


### Why Tabular Methods Do Not Scale

Tabular methods suffer heavily from the _Curse of Dimensionality_. If we wanted to increase the resolution of our discretization to 10 bins per variable to achieve finer control, the state space would grow exponentially to $10^4 = 10,000$ states. If we moved to a slightly more complex environment like LunarLander-v2 (an 8-dimensional state space) with 10 bins, the table would require $10^8$ entries. 

Because Tabular Q-learning must visit every state-action pair multiple times to converge to an optimal policy, it becomes computationally intractable—and hopelessly memory-inefficient—to train in high-dimensional continuous environments. Furthermore, tabular methods lack generalization; learning the Q-value for one bin provides absolutely no information about the neighboring bins.

## Task 2: DQN

The following figure shows a graph of three DQN variants:
1. the DQN-noTarget, the green line, with replay buffer, but not target network
2. the DQN-noReplay, the red line, with no replay buffer, but with target network
3. the DQN-full, the blue line, which has replay buffer and target network

<img src="results/task2_dqn_ablation.png" 
     alt="DQN ablation study graph"
     width="700" />

The DQN-noTarget variant completely fails to learn and stays flat near zero.

This behaviour captures the "Moving Target" problem. Without a frozen Target Network, the agent is attempting to learn using a target 

$Y = r + \gamma \max_{a'} Q_{\theta}(s', a')$

where the parameters $\theta$ are being updated every single step.

This leads to the situation where the Q-values oscillate greatly or diverge. In this specific experiment, the lack of a stable target likely caused the Q-values to collapse or grow uncontrollably, leading to the agent never discovering a successful policy for balancing the pole. It effectively did not find a way of how to learn from the beginning.


The variant of DQN-noReplay actually shows surprisingly high returns initially, but it eventually exhibits  declining behaviour after 42000 steps. This is caused by removing the Replay Buffer. The agent is forced to learn from consecutive transitions ($s_t, s_{t+1}, s_{t+2} \dots$). These transitions are highly correlated.
However, neural networks are designed for i.i.d. (independent and identically distributed) data.

Therefore, the network starts to show overfitting to the immediate local experience. 
The line shows an initial upward trend hitting a peak, and then it starts to drop in a high decreasing rate. This is because the agent is learning only from its most recent, highly correlated experience, causing it to lose information about the strategies it learned in the past. It effectively "forgets" how to handle earlier states because it is no longer being reminded of them by the Replay Buffer.

Finally, we note the behaviour of the DQN-full bariant, with the blue line.
The model shows an inceaing trend as the DQN-noReplay variant and then it show a descreasing behaviour in results ar around 10000 steps, then it recovers at 16000 steps and then it demontstrate a sharp drop almost vertical at around 27,000 steps.

Why this might happen: In many DQN implementations on CartPole, this can happen if the network overfits to a specific "local" strategy or if the $\epsilon$-greedy exploration decayed too fast, causing the agent to get stuck in a "bad" policy that it can no longer escape.

This study shows that even with the full DQN components, reinforcement learning is highly sensitive. You could mention that this highlights the need for further tuning (e.g., using a larger replay buffer, different learning rates, or double DQN).


### some more investigation on e_decay_Steps

For a faster or slower decay we uncover different behaviours. Most interestingly for `e_decay_steps=5000` (line 19 file `experiments/hyperparameter_task2/task2_experiment_dqn_e_decay_steps.py`) we get the following image, wher the graph show again that the variant with NO replay has a better behaviour

<img src="results_e_slow/task2_dqn_ablation.png" 
     alt="DQN ablation study graph"
     width="700" />

However, in this case the full model overtakes the noReplay in the region of steps 20-35 thousnad but then the noReplay is becoming better after 35 thousand steps. However, both models at 50 thousand steps are degrading at the same rate.

A possible explanation for this finding is that the DQN-noReplay variant (the red line) is essentially performing online learning and because it is learning from the most recent, immediate experience, it can sometimes converge to a "good enough" policy much faster than the DQN-full version, which has to wait to sample its experiences from a large, diverse Replay Buffer.

The Replay Buffer (in DQN-full) acts as a "stabilizer" that prevents the network from overfitting. While this may be crucial for stability in complex environments (like Atari or LunarLander), in a simple, low-dimensional environment like CartPole, the Replay Buffer might actually be "slowing down" the learning by diluting the updates with older, less relevant data.

This shows that more investigation is needed to understand what is happeing in the training and highlights the need for this type of evaluations such as the ablastion study.

It will be fruitful to see this hyperparameter study in a different environment in task 4b (time permitted) 

On the other hand, with a faster decaying $\epsilon$ the DQN-noReplay variant ouperforms the DQN full variant at all steps as shown in the image below.

<img src="results_e_fast/task2_dqn_ablation.png" 
     alt="DQN ablation study graph"
     width="700" />

In all cases the variant DQN-noTarget has the worst outcome showning clearly the need for the implementation of the Target network, as the agent never understands how to start learning.

## Final thoughts on the ablaition study

From the findings on the Ablation study we confirm the necessity of the architectural components of DQN.

The DQN-noTarget variant fails completely, validating that a stable, frozen target is required to prevent the divergence of the Q-function. The DQN-noReplay variant demonstrates the danger of catastrophic forgetting; while it initially learns, the lack of sample decorrelation via a Replay Buffer prevents it from maintaining a stable policy, leading to performance degradation. 

(_Disclaimer_, note from Gemini3.5 Pro, but paper is marked in README for further investigation): These results align with the foundational findings by Mnih et al. (2015), confirming that stable learning in high-dimensional environments requires both stationary targets and uncorrelated experience sampling.

## Evaluation Metric: Environment Steps vs. Episodes

While conducting the study the question which metrics we use to compare the models in the same scale arises. 

Throughout this empirical study, algorithmic performance is plotted against **total environment steps** rather than episodes or gradient updates. An environment step is defined as a single transition ($s_t \rightarrow s_{t+1}$) experienced by the agent. Measuring against environment steps standardizes the x-axis around **sample efficiency**. 

For DQN, the relationship between environment steps and training steps is essentially 1:1.

Once the initial warmup phase is complete, DQN operates on a continuous, step-by-step interleaving process. During a single iteration of the main training loop:
- Environment Step: The agent looks at the state, chooses an action, and the environment moves forward by one frame. The agent receives the reward and next state. (+1 Environment Step)
- Storage: That single experience is pushed into the Replay Buffer.
- Training Step: The agent immediately grabs a random batch of 64 experiences from the Replay Buffer and performs gradient descent to update its neural network weights. (+1 Training Step)

Because these two actions are locked together inside the `while not done:` loop, a DQN agent that has experienced 100,000 environment steps has also performed exactly 100,000 training steps (minus the warmup buffer).

This makes DQN highly sample efficient (it learns a lot from a small amount of environment interaction) but very computationally expensive (it runs a full neural network backpropagation for every single step it takes in the game).

However, algorithms like REINFORCE only perform one network update per episode, and episodes vary wildly in length during early training, comparing algorithms by episodes or training steps would unfairly skew the learning curves.

## Task 3: Policy Gradient

### A. REINFORCE

The requirements from the exercise relate to the following points:

1. Policy Network: Instead of outputting Q-values, the network outputs preferences for each action, which are passed through a Softmax function. This turns them into probabilities (e.g., 70% chance to move left, 30% chance to move right
2. Return Computation (Monte Carlo): The agent plays a full episode from start to finish before learning anything. Once the episode is over, it looks backward. For every step $t$, it calculates the exact discounted future return $G_t = \sum^{T−t−1}_{k=0} \gamma^{k}r_{t+k+1}$.
3. Return Normalization: Raw returns cause highly unstable gradients. By subtracting the mean and dividing by the standard deviation of the returns within that specific episode, we create a "baseline." Actions that performed better than average get a positive score, and actions that performed worse than average get a negative score.
4. Gradient Update: We minimize the loss $L = -E[G_t \cdot \log\pi_\theta(a_t|s_t)]$;  if an action resulted in a positive normalized return, the gradient pushes the network to increase the probability of taking that action again. If the return was negative, the network is pushed to decrease that probability.

#### Results

The following image show the graph of the returns vs the environment steps for the REINFORCE algorithm

<img src="results/task3_reinforce.png" 
     alt="DQN ablation study graph"
     width="700" />

At a first glance, we notice a volatile behaviour throughout the experiment with very high standard deviation. Also we notice that the experiment reaches up to  275000 steps.

From these observations, we may conclude that the learning curve suffers from severe instability and high variance, evidenced by the wide standard deviation bands and catastrophic performance drops at around 60, 125 and 200 thousand steps (which spikes up at around 80 and 140 thousand steps).

Furthermore, REINFORCE proves to be highly sample-inefficient compared to DQN, requiring over 250,000 environment steps to approximate convergence. This instability highlights the difficulty of relying on raw, high-variance episodic returns for gradient updates without a learned baseline or value function.

This behaviour stems from the fact that REINFORCE is a Monte Carlo method. It updates its network based on the total return of a full episode. If the agent takes 400 good steps and 1 terrible step that ends the game, it might penalize all 401 steps. This makes the gradient estimates highly noisy and creates massive variance between different training runs.

### B. A2C

#### Results


## Task 4: Comparison of all agents and Hyperparameter stydy

### A. Comparison

#### Plot all agents.

#### Table: Comparison of all agents on their results

|Algorithm | Steps to solve | Final return (mean)| Std  | Wall-clock (min)|
|----------|----------------|--------------------|------|-----------------|
|Tabular Q-learning |  Did not solve  | 264.3    |3,45  | 0.28            | 
|DQN                |  Did not solve  | 105.42   |28.24 | 1.27            | 
|REINFORCE          |  Did not solve  | 348.3    |105.77| 3.07            |
|A2C                |       |                    |      |                 |

_Note_: results for each model are in the files `(agent)_comparing_table.csv`

#### Discussion

### Hyperparameter Study


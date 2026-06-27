# AI HAnds-on 3rd assignemnt - Reinforcement learning

## Task 1: Tabular Q-Learning Baseline

Q-Table Size: To apply tabular Q-learning to the continuous CartPole-v1 environment, the 4-dimensional state space was discretized into 6 uniform bins per variable. This results in $6^4 = 1296$ unique discrete states. Because there are 2 possible actions, the **final Q-table size** is $1296 \times 2 = 2592$ state-action pairs.

### Why Tabular Methods Do Not Scale

Tabular methods suffer heavily from the _Curse of Dimensionality_. If we wanted to increase the resolution of our discretization to 10 bins per variable to achieve finer control, the state space would grow exponentially to $10^4 = 10,000$ states. If we moved to a slightly more complex environment like LunarLander-v2 (an 8-dimensional state space) with 10 bins, the table would require $10^8$ entries. 

Because Tabular Q-learning must visit every state-action pair multiple times to converge to an optimal policy, it becomes computationally intractable—and hopelessly memory-inefficient—to train in high-dimensional continuous environments. Furthermore, tabular methods lack generalization; learning the Q-value for one bin provides absolutely no information about the neighboring bins.


## Task 4

Table

|Algorithm | Steps to solve | Final return (mean)| Std | Wall-clock (min)|
|----------|----------------|--------------------|-----|-----------------|
|Tabular Q-learning |  did not solve  |         264.3      |3,45 |        0.28     | 
|DQN                |       |                    |     |                 | 
|REINFORCE          |       |                    |     |                 |
|A2C                |       |                    |     |                 |

_Note_: results for each model are in the files `agent_comparing_table.csv`
# Self-Driving Car

The repository presents the results of the research project implemented in the framework of scientific collaboration with [Arrival](https://arrival.com/world/en). The global objective of the research project is to investigate the ability of using Reinforcement Learning (RL) for solving various problems concerned with Self-Driving Car. The car model, provided by Arrival, describes the dynamics of a real car in detail and, in particular, take into account the tire models describe the forces in the road-tire interactions according to the dynamic condition of the vehicle. The repository describes only the results of experiments, but does not include the model itself. 

**Implementation:** Vitaly Kalev, Aleksandr Goranov, Anton Plaksin

**Advisors:** Vladimir Bulaev


## Task 1

The vehicle dynamics are influenced by

- longitudinal tire forces, 
- aerodynamic drag forces,
- rolling resistance forces,
- gravitational forces.

For example, forces and moments from the road act on each tire of the vehicle and highly influence the dynamics of the vehicle. If the longitudinal slip ratio is not small or if the road is slippery, then a nonlinear tire model needs to be used to calculate the longitudinal tire force.  Algorithms of car control systems use information about the tire-road friction coefficient and tuned to work with dry, wet or icy coating. Therefore, having information about this coefficient value allows to subsequently cost control in a more effectively. 

The problem can be formalized as one-step Markov Decision Process (MDP) $(A,R)$. Here $A = [\nu_{min}, \nu_{max}]$ is the interval of admissible values of $\nu$ and 

$R = \int\limits_0^1\bigg( \sum\limits_{i \in I} |\hat{x}_i(t) - x_i(t)|\bigg) d t$

is the reward function, where $I$ is the set of indexes of observed coordinates, $\hat{x}$ is the realized motion of the car with unknown $\hat{\nu}$, and $x_i$ is the motion of the car model dependent on choice of $\nu$. We aims to find $\nu_0$ such that $R = 0$ and verify the equality $\nu_0 = \hat{\nu}$.

To solve the problem, we use cross-entropy method (CEM). Its parameters and other details of the experiment can be found in ???.

Pic???

## Task 2

Since a real car and its model can be different, it is important to study the stability issue of the algorithm performance iwth respect to the car parameters (such as length, mass, tire parameters, etc.). To investigate this, 

???

## Task 3

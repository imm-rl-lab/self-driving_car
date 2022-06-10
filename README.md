# Self-Driving Car

The repository presents the results of the research project implemented in the framework of scientific collaboration with [Arrival](https://arrival.com/world/en). The global objective of the research project is to investigate the ability of using Reinforcement Learning (RL) for solving various problems concerned with Self-Driving Car. The car model, provided by Arrival, describes the dynamics of a real car in detail and, in particular, take into account the specifics of tires. The repository describes only the results of experiments, but does not include the model itself.

## Task 1

Динамика автомобля зависит от начально потолжения, его управления и нескольких неизвенствных параметров. Одним из таких параметров является ???. Знание этого параметра позволяет в последствии стоить управление более экстремальным образом. Таким образом, возникает задача о поиске этого параметра при по зная скорость и наблюдаемые координаты траектории. 

The problem can be formalized as one-step Markov Decision Process $(A,R)$. Here $A = [\nu_{min}, \nu_{max}]$ is the interval of admissible values of $\nu$ and 

$R = \int\limits_0^1\bigg( \sum\limits_{i \in I} |\hat{x}_i(t) - x_i(t)|\bigg) d t$

is the reward function, where $I$ is the set of indexes of observed coordinates, $\hat{x}$ is the realized motion of the car with unknown $\hat{\nu}$, and $x_i$ is the motion of the car model dependent on choice of $\nu$. We aims to find $\nu_0$ such that $R = 0$ and verify the equality $\nu_0 = \hat{\nu}$.

To solve the problem, we use CEM algorithm. Its parameters and other details of the experiment can be found in ???.

Pic???

## Task 2

Since a real car and its model can be different, it is important to study the stability issue of the algorithm performance iwth respect to the car parameters (such as length, mass, tire parameters, etc.). To investigate this, 

???

## Task 3

# Snake-Ai

A machine learning model that learns to play the game "Snake".

### Game Actions

[1, 0, 0] -> straight
[0, 1, 0] -> right turn
[0, 0, 1] -> left turn

### Rewards

eat food: +10
game over: -10
else: 0

### Game State

At any given moment, our model needs to know these 11 state variables.
[
danger straight, danger right, danger left,
direction left, direction right,
direction up, direction down,
food left, food right,
food up, food down
]

### Model Structure

#### Input Layer

The input layer is comprised of 11 neurons. Each neuron is associated with one of the state variables.

#### Output Layer

The output layer is comprised of 3 neurons. The 3 neurons will generate a game action such [1, 0, 0] -> straight.

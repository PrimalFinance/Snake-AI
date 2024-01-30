import torch
import random
import numpy as np
from collections import deque
# Import model 
from Model.model import Linear_QNet, QTrainer
# Snake game imports
from Game.snake_game import SnakeGameAI, Direction, Point

# Utils import 
from Utils.graphs import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001

class Agent:
    def __init__(self) -> None:
        self.num_games = 0
        self.epsilon = 0 # Controls randomness 
        self.gamma = 0.9 # Discount rate. Must be smaller than 1.
        self.memory = deque(maxlen=MAX_MEMORY) # Create deque data structure. When memory exceeds max length, it will remove elements from the left (popleft()).
        # Input size: 11 due to tracking 11 features. 
        # Hidden layer: 256
        # Output layer: 3 since, the model is creating a coordinate [0, 1, 0] as an example. 
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    ###########################
    # 
    #
    #
    ###########################

    def get_state(self, game):
        head = game.snake[0] # First item in the list is head of the snake. 
        # Create 4 point around the head of the snake. Intervals of 20 are used since that is "BLOCK_SIZE" in the game file. 
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        # Boolean to determin which direction the current game direction is equal to. 
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # The state expressions require 2 pieces of information. 
        # The current direction, and the points surrounding the snakes head. 
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or # This is considered "straight" because, if the snake turned right, point_r which represents the coordinate to the right of the snake's head before the move, is now straight in front of the head, after the move.  
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left of snake head
            game.food.x > game.head.x,  # food right of snake head
            game.food.y < game.head.y,  # food up 
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
    ###########################
    # 
    #
    #
    ###########################
    def remember(self, state, action, reward, next_state, game_over_state):
        # Convert all the params into 1 tuple, so technically 1 element is appended, rather than 5.
        self.memory.append((state, action, reward, next_state, game_over_state)) # Popleft if MAX_MEMORY is reached. 
    ###########################
    # 
    #
    #
    ###########################
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Create a mini-sample from a random selection of the models memories, of size BATCH_SIZE.
        else:
            # If our memory is less than BATCH_SIZE, then the current memories can be used for the sample. 
            mini_sample = self.memory
            
        states, actions, rewards, next_states, game_over_states = zip(*mini_sample)
        # NOTE: These variables are plural because each variable is a collection of the field. Ex: Instead of one state, states contains information about multiple states.  
        self.trainer.train_step(states, actions, rewards, next_states, game_over_states)
       
            
    ###########################
    # @dev Train the model on one step.
    #
    #
    ###########################
    def train_short_memory(self, state, action, reward, next_state, game_over_state):
        self.trainer.train_step(state, action, reward, next_state, game_over_state)
    ###########################
    # 
    #
    #
    ###########################
    def get_action(self, state):
        # Random moves: tradeoff exploration / exploitation. 
        # The more games that are played, the smaller the epsilon becomes. 
        # This is wanted because, when the model first starts we want some form of randomness (high epsilon). 
        # However once, the model has learned, we can remove this randomness (lower epsilon) and have it create it's own moves. 
        self.epsilon = 80 - self.num_games
        final_move = [0, 0, 0]
        # The lower the epsilon gets, the less likely this if statement will return true. 
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2) # Gives random integer between 0 and 2
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move
        
        

def train():
    plot_scores = []
    plot_mean_scores = [] # Average scores
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        # Retrieve the current state. 
        state_old = agent.get_state(game)
        # Get move 
        final_move = agent.get_action(state_old)
        # Perform move and get new state. 
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Train short memory (1 step)
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)
        agent.remember(state_old, final_move, reward, state_new, game_over)
        
        # If the game has ended. 
        if game_over:
            # Train long memory // Aka replay memory
            # Plot results. 
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
            
            print(f"Game: {agent.num_games}\nScore: {score}\nRecord: {record}")
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()


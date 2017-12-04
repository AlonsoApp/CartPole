from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, Adagrad
import numpy as np
import gym
import pickle
from collections import deque
import random
import progressbar

TOTAL_GAMES = 1000
GAME_MAX_STEPS = 500
INPUT_SIZE = 4
ACTION_SIZE = 2

class Agent:
    def __init__(self, hidden_layer_size, epsilon_rate, gamma, batch_size, optimizer):
        self.optimizer = optimizer
        self.hidden_layers = 2
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = 0.001 # Rate in which the loss function corrects old values to the new ones
        self.epsilon = 1.0 # defines initial exploitation v exploration ratio
        self.epsilon_rate = epsilon_rate # how much epsilon will decrement every learning step
        self.action_size = ACTION_SIZE # number of actions the agent can perform
        self.input_size = INPUT_SIZE # size of the input layer
        self.gamma = gamma  # discount rate how much we want to propagate future q values across prev states
        self.min_epsilon = 0.01 # min val of eps to keep taking some exploration actions
        self.memory = deque(maxlen=100000) # list of (state, action, reward, next_state, done) for the long-term memory
        self.batch_size = batch_size # size of the batch we take from memory to relearn
        # stored to do the learning after every play
        self.game_log = []
        self.model = self.create_nn()

    # Creates the NN used as q function
    def create_nn(self):
        model = Sequential()

        model.add(Dense(self.hidden_layer_size, input_dim=self.input_size, activation='relu'))

        # We parametrice the amount of hidden layers for teh refinement process
        for _ in range(1, self.hidden_layers):
            model.add(Dense(self.hidden_layer_size, activation='relu'))

        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self.optimizer(lr=self.learning_rate))
        return model

    # Returns an action for the given state
    def action(self, state):
        if np.random.rand() > self.epsilon:
            # Exploitation
            action = np.argmax(self.model.predict(state)[0])
        else:
            # Exploration
            action = np.random.randint(0, self.action_size)
        return action

    # Retrains the model with just one (state, action, reward, next_state, done)
    # Not used anymore since long term memory was added "learn_from_memory"
    def learn(self, state, action, reward, next_state, done):
        new_state_q_action = reward
        if not done:
            new_state_q_action = reward + self.gamma * max(self.model.predict(next_state)[0])
        state_q = self.model.predict(state)
        # we update the q value for the taken action on this state to the updated q value of the next state
        state_q[0][action] = new_state_q_action
        self.model.fit(state, state_q, batch_size=1, verbose=0)

    # Gets random batch_size sized batches form the (state, action, reward, next_state, done) memory
    def generate_learning_batches(self):
        state_batch, state_q_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        for state, action, reward, next_state, done in minibatch:
            new_state_q_action = reward
            if not done:
                new_state_q_action = reward + self.gamma * max(self.model.predict(next_state)[0])
            state_q = self.model.predict(state)
            # we update the q value for the taken action on this state to the updated q value of the next state
            state_q[0][action] = new_state_q_action
            state_batch.append(state[0])
            state_q_batch.append(state_q[0])
        return state_batch, state_q_batch

    # Learns from memory and fits the model to provide the agent with long-term memory
    def learn_from_memory(self):
        # get random set of states <-> q_target for these states
        state_batch, state_q_batch = self.generate_learning_batches()

        self.model.fit(np.array(state_batch), np.array(state_q_batch), batch_size=len(state_batch), verbose=0)

    # Decreases epsilon every play if epsilon is not below the limit
    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon - (self.epsilon*self.epsilon_rate)

    # Logs the result of a game (score, epsilon, avg) to later generate the learning progress graph
    def log(self, game, total_games, score, epsilon):
        hundred_avg = -1
        if game/100 > 0:
            tras = zip(*self.game_log)
            data = list(tras[2])
            hundred_avg = np.average(data[-100:])
        print("Game: {}/{} Steps: {} epsilon: {:.2} avg: {}"
              .format(game, total_games, score, epsilon, hundred_avg))
        self.game_log.append((game, total_games, score, epsilon, hundred_avg))

    # Saves all the logs saved during the learning process on a pickle
    def save_log(self, punishment):
        name = "./pickles/{}_er{}_g{}_{}_{}_{}.p".format(self.hidden_layer_size, self.epsilon_rate, self.gamma, punishment, self.batch_size, str(self.optimizer.__name__))
        pickle.dump(self.game_log, open(name, "wb"))

    # Generates a height x width x 2 matrix with the q values of each action at every cart_position x pole_position
    def q_matrix(self, height, width):
        print("Generating Q-Matrix")
        cart_min, cart_max, cart_samples = -2.4, 2.4, height
        cart_axis = np.linspace(cart_min, cart_max, num=cart_samples)
        pole_min, pole_max, pole_samples = -0.26, 0.26, width
        pole_axis = np.linspace(pole_min, pole_max, num=pole_samples)
        matrix = [[None for _ in range(len(pole_axis))] for _ in range(len(cart_axis))]
        with progressbar.ProgressBar(max_value=len(cart_axis)) as bar:
            for i, cart_val in enumerate(cart_axis):
                bar.update(i)
                for j, pole_val in enumerate(pole_axis):
                    state = np.reshape([cart_val, 0, pole_val, 0], [1, INPUT_SIZE])
                    tmp = self.model.predict(state)[0]
                    matrix[i][j] = tmp
        return matrix

    # Loads the weights form a h5 file
    def load_weights(self, file_name):
        self.model.load_weights(file_name)

    # Saves the weights in a h5 file
    def save_weights(self, punishment):
        name = "./weights/{}_er{}_g{}_{}_{}_{}.h5".format(self.hidden_layer_size, self.epsilon_rate, self.gamma, punishment, self.batch_size, str(self.optimizer.__name__))
        self.model.save_weights(name)

    # Prints on console an easy to read 2D matrix
    def print_matrix(self, matrix):
        for row in matrix:
            str = ""
            for val in row:
                str += "[{:.2}]".format(val)
            print(str)

    # Converts a two action matrix into an greyscale image for teh given action
    def draw_matrix(self, matrix, action):
        x = np.array(matrix)[:, :, action]
        #self.print_matrix(x)
        # we normalize regarding to both action values
        m_max = np.array(matrix).max() #x.max()
        m_min = np.array(matrix).min() #x.min()
        z = (x - m_min) / (m_max - m_min)
        pix = z * 255

        from PIL import Image
        im = Image.fromarray(pix)
        im.show()

    # Greyscale image of left - right q-matrix
    def dif_matrix(self, matrix):
        x = np.array(matrix)[:, :, 0]
        y = np.array(matrix)[:, :, 1]
        x += np.abs(x.min())
        y += np.abs(y.min())
        m = np.subtract(x, y)

        # we normalize regarding to both action values
        m_max = np.array(m).max() #x.max()
        m_min = np.array(m).min() #x.min()
        z = (m - m_min) / (m_max - m_min)
        pix = z * 255

        from PIL import Image
        im = Image.fromarray(pix)
        im.show()

    # Shows two grey scale images representing the q matrix for each action at every cart_position x pole_position
    def show_q_matrix(self):
        matrix = agent.q_matrix(100, 100)
        self.draw_matrix(matrix, 0)
        self.draw_matrix(matrix, 1)
        #self.dif_matrix(matrix)

def play_learn(env, agent, finish_reward):
    for i in range(TOTAL_GAMES):
        state = env.reset()
        # we reshape the state [4] because the model needs it in a [1, 4] shape tensor
        state = np.reshape(state, [1, INPUT_SIZE])
        steps = 0
        done = False
        while not done and steps < GAME_MAX_STEPS:
            # env.render()
            # choose an action given a state
            action = agent.action(state)
            # perform the action and get feedback form env
            next_state, reward, done, _ = env.step(action)
            # we reshape the state [4] because the model needs it in a [1, 4] shape tensor
            next_state = np.reshape(next_state, [1, INPUT_SIZE])
            # if the game done we use finish_rewrd as reward
            reward = reward if not done else finish_reward
            # add this experience to the agent memory
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            steps += 1

        agent.log(i, TOTAL_GAMES, steps, agent.epsilon)
        # Once the game has ended, train the model with random experiences form its memory
        agent.learn_from_memory()

        # we decrease epsilon each game
        agent.update_epsilon()
    # we save learning progress
    agent.save_log(finish_reward)
    # we save the weights
    agent.save_weights(finish_reward)

def play_demo(env, agent, finish_reward):
    agent.epsilon = 0.00
    for i in range(TOTAL_GAMES):
        state = env.reset()
        # we reshape the state [4] because the model needs it in a [1, 4] shape tensor
        state = np.reshape(state, [1, INPUT_SIZE])
        steps = 0
        done = False
        while not done and steps < GAME_MAX_STEPS:
            env.render()
            # choose an action given a state
            action = agent.action(state)
            # perform the action and get feedback form env
            next_state, reward, done, _ = env.step(action)
            # we reshape the state [4] because the model needs it in a [1, 4] shape tensor
            next_state = np.reshape(next_state, [1, INPUT_SIZE])
            state = next_state
            steps += 1

        agent.log(i, TOTAL_GAMES, steps, agent.epsilon)
    # we save learning progress
    agent.save_log(finish_reward)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')

    agent = Agent(16, 0.005, 0.9, 32, Adam)
    agent.load_weights("./weights/16_er0.005_g0.9_-20_32_Adam.h5")
    agent.show_q_matrix()
    play_demo(env, agent, -20)
    # If you want to make the agent learn comment above line and use this instead
    # play_learn(env, agent, -20)

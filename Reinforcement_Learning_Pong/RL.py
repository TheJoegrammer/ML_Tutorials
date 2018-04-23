#1) Get neccesary imports
from __future__ import print_function
import os.path
import numpy as np
import tensorflow as tf
import argparse
import gym

#Set the network information.
OBSERVATIONS_SIZE = 210*160
hidden_layer_size=200
learning_rate=0.05
batch_size_episodes=1
checkpoint_every_n_episodes=10
load_checkpoint='true'
discount_factor=0.99
render='true'


#2) Initialize AI-GYM specific action values to send to gym environment to move paddle up/down
UP_ACTION = 4
DOWN_ACTION = 5
#3) Map action values to outputs that'll impact the policy network.
action_dict = {DOWN_ACTION: -1, UP_ACTION: 1}

#4) Resize matrix from 210x160x3 to 33600, Taking out 3rd dimension
##Inputs:
    #Uint8 frame
#Outputs:
    #The processed image (33600 element vector, of 0's (background) or 1's (ball and paddle)
def preprocess_image(image):
    """ preprocess the 210x160x3 uint8 frame into 33600 (210x160) 1D float vector """
    image = image[:, :, 0]  # Take out 3rd dimension
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return image.astype(np.float).ravel()

#5) Discount the reward achieved by the gradient.
##This makes it such that the events taken at the end of the game are weighted more heavily than the ones at the beginning.
##This sets up before we use backprop to compute the gradient.
##Inputs:
    #Rewards:
#TODO: Check if discount_rewards does reward per movement? or reward per game? If game, understandable but why does commented part exisst? If movement, what-even-tho.
def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)#Make a 0's array with the same size and type as rewards.
    for t in range(len(rewards)):#For each row...
        discounted_reward_sum = 0#Sum up all of the rewards that have been discounted
        discount = 1#Initialize discount as "1" for highest discount.
        for k in range(t, len(rewards)):#Go from current row->end of rows
            #Sum the total discounted reward by adding rewards*discount
            discounted_reward_sum += rewards[k] * discount
            #Discount is then multiplied by discount factor, to make discount smaller.
            discount *= discount_factor
            ##TODO: MIGHT BE WEAK CODE...
            ##If you have a succesful run...
            #if rewards[k] != -1:#COULD BE 0?
            #    # Don't count rewards from subsequent rounds! Might as well not add a bunch of weak rounds.
            #    break
        #Make a new discounted-rewards, where the larger rewards are more important!
        discounted_rewards[t] = discounted_reward_sum
    return discounted_rewards

#5.5) Initialize the network the network.
class Network:
    def __init__(self, hidden_layer_size, learning_rate, checkpoints_dir):
        #Make the session
        self.sess = tf.InteractiveSession()
        #Get your observation information, in a batch the size of your observation, size of the image.
        self.observations = tf.placeholder(tf.float32,
                                           [None, OBSERVATIONS_SIZE])
        # +1 for up, -1 for down.
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        #Get the advantage matrix. Which is used for coefficient for loss. 
        self.advantage = tf.placeholder(
            tf.float32, [None, 1], name='advantage')
        #Get a densely connected matrix
        h = tf.layers.dense(
            self.observations,#that takes in the observations as inputs
            units=hidden_layer_size,#Dimensionality of output space
            activation=tf.nn.relu,#Activation function passed into layer, so each item in output space passes through relu
            kernel_initializer=None)#Initializer function for weight matrix, TODO check into this?
        #Make a second densley connected matrix.
        self.up_probability = tf.layers.dense(
            h,#Take in your previous layer as a densly connected layer.
            units=1,#Dimensionality of output space
            activation=tf.sigmoid,#Activation function passed into layer, so each item passed through sigmoid
            kernel_initializer=None)#Initializer function for weight matrix, TODO check into this?

        #Now it's time to train the network!
        #   Do so based on log proabbility of the sampled action.
        #In order to encourage actions that we did when the agent won, and discourage actions where
        #agent lost, we'll increase log probability of winning actions and drecrease probability of losing actions.
        #
        # Direction to push log probability controlled by "advantage", reward for each action in each round.
        # Positive reward pushes log probability of chosen action up, negative reward pushes action probability down.
        
        #Initialize logLoss
        self.loss = tf.losses.log_loss(#Use log-losses
            labels=self.sampled_actions,#Actions: Where amount taken are # in sampled_actions.
            predictions=self.up_probability,#Scalar probability that the paddle will go up. 
            weights=self.advantage)#Each loss rescaled by whatever wight value is in that current element.

        #Optimizer: Implements the Adam Algorithm.
        optimizer = tf.train.AdamOptimizer(learning_rate)

        #the training operation specified is the Adam optimizer minimizing the log-loss.
        self.train_op = optimizer.minimize(self.loss)

        #tf.global_variables_initializer() initilaizes global variables in TF graph
            #runs the graph
        tf.global_variables_initializer().run()

        #Saves the thing.
        self.saver = tf.train.Saver()
        #Make where you want your data saved as a 
        self.checkpoint_file = os.path.join(checkpoints_dir,
                                            'policy_network.ckpt')
    #Following 2 functions are for saving and loading checkpoints
    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    #Find the probability to go up
    #Output:
            #self.sess.run returns the first argument.
            #The up_proabbility is the scalar at the end of the forward-passing network.
            #The feed_dict argument takes in the observations, reshaped observations.
    def forward_pass(self, observations):
        up_probability = self.sess.run(
            self.up_probability,
            feed_dict={self.observations: observations.reshape([1, -1])})
        return up_probability

    #Whatever works
    def train(self, state_action_reward_tuples):
        print("Training with %d (state, action, reward) tuples" %
              len(state_action_reward_tuples))

        states, actions, rewards = zip(*state_action_reward_tuples)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        feed_dict = {
            self.observations: states,
            self.sampled_actions: actions,
            self.advantage: rewards
        }
        self.sess.run(self.train_op, feed_dict)

#6) Make a Pong game
env = gym.make('Pong-v0')

#7) Make a network, take in hidden layer size, hidden layer rate, and checkpoints.
network = Network(hidden_layer_size, learning_rate, checkpoints_dir='checkpoints')
#8) If you HAVE a checkpoint, load it. Please.
if load_checkpoint=='true':
    network.load_checkpoint()

#9) Get together all of the reward tuples, smooth reward and get episode #
batch_state_action_reward_tuples = []
smoothed_reward = None
episode_n = 1

#10) Begin this training.
while True:
    #Get the episode #
    print("Starting episode %d" % episode_n)
    #Set episode information
    episode_done = False
    episode_reward_sum = 0
    round_n = 1
    
    #Set environment, preprocess it, 
    last_observation = env.reset()
    last_observation = preprocess_image(last_observation)
    action = env.action_space.sample()
    observation, _, _, _ = env.step(action)
    observation = preprocess_image(observation)
    n_steps = 1

    while not episode_done:
        if render=='true':
            env.render()

        observation_delta = observation - last_observation
        last_observation = observation
        up_probability = network.forward_pass(observation_delta)[0]
        if np.random.uniform() < up_probability:
            action = UP_ACTION
        else:
            action = DOWN_ACTION

        observation, reward, episode_done, info = env.step(action)
        observation = preprocess_image(observation)
        episode_reward_sum += reward
        n_steps += 1
        tup = (observation_delta, action_dict[action], reward)
        batch_state_action_reward_tuples.append(tup)

        if reward == -1:
            print("Round %d: %d time steps; lost..." % (round_n, n_steps))
        elif reward == +1:
            print("Round %d: %d time steps; won!" % (round_n, n_steps))
        if reward != 0:
            round_n += 1
            n_steps = 0

    print("Episode %d finished after %d rounds" % (episode_n, round_n))

    # exponentially smoothed version of reward
    if smoothed_reward is None:
        smoothed_reward = episode_reward_sum
    else:
        smoothed_reward = smoothed_reward * 0.99 + episode_reward_sum * 0.01
    print("Reward total was %.3f; discounted moving average of reward is %.3f" \
        % (episode_reward_sum, smoothed_reward))

    if episode_n % batch_size_episodes == 0:
        states, actions, rewards = zip(*batch_state_action_reward_tuples)
        rewards = discount_rewards(rewards, discount_factor)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)
        batch_state_action_reward_tuples = list(zip(states, actions, rewards))
        network.train(batch_state_action_reward_tuples)
        batch_state_action_reward_tuples = []

    if episode_n % checkpoint_every_n_episodes == 0:
        network.save_checkpoint()

    episode_n += 1

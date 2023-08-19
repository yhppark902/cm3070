from abc import ABC, abstractmethod

# from environment import BaseEnvironment
import numpy as np

# class PendulumEnvironment(BaseEnvironment):
class PendulumEnvironment(ABC):


    def __init__(self):
        self.rand_generator = None
        self.ang_velocity_range = None
        self.dt = None
        self.viewer = None
        self.gravity = None
        self.mass = None
        self.length = None
        
        self.valid_actions = None
        self.actions = None
        
        self.playfirst = None
        self.playrandom = None
    
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        
        Set parameters needed to setup the pendulum swing-up environment.
        """
        # set random seed for each run
        self.rand_generator = np.random.RandomState(env_info.get("seed"))     
        
        self.ang_velocity_range = [-2 * np.pi, 2 * np.pi]
        self.dt = 0.05
        self.viewer = None
        self.gravity = 9.8
        self.mass = float(1./3.)
        self.length = float(3./2.)
        
        self.valid_actions = (0,1,2,3,4)
        self.actions = [-2,-1,0,1,2]
        
        self.last_action = None
        
        self.playfirst = True
        self.playrandom = False

    
    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

        ### set self.reward_obs_term tuple accordingly (3~5 lines)
        # Angle starts at -pi or pi, and Angular velocity at 0.
        # reward = ?
        # observation = ?
        # is_terminal = ?
        
        beta = -np.pi
        betadot = 0.
        
        reward = 0.0
        observation = np.array([beta, betadot])
        is_terminal = False
        
        self.reward_obs_term = (reward, observation, is_terminal)
        
        # return first state observation from the environment
        return self.reward_obs_term[1]
    
    @abstractmethod
    def env_step(self, action):
        pass
    
    def ode_solver(self,action):
        
        assert(action in self.valid_actions)
        
        last_state = self.reward_obs_term[1]
        last_beta, last_betadot = last_state        
        self.last_action = action
        
        betadot = last_betadot + 0.75 * (self.actions[action] + self.mass * self.length * self.gravity * np.sin(last_beta)) / (self.mass * self.length**2) * self.dt

        beta = last_beta + betadot * self.dt

        # normalize angle
        beta = ((beta + np.pi) % (2*np.pi)) - np.pi
        
        # reset if out of bound
        if betadot < self.ang_velocity_range[0] or betadot > self.ang_velocity_range[1]:
            beta = -np.pi
            betadot = 0.
        
        # compute reward
        reward = -(np.abs(((beta+np.pi) % (2 * np.pi)) - np.pi))
        observation = np.array([beta, betadot])
        is_terminal = False
        
        return reward,observation,is_terminal
    def play_last(self):
        self.playfirst = False
        return None
    def play_first(self):
        self.playfirst = True
        return None
    def play_random(self):
        self.random = True
        return None
    def play_learner(self):
        self.random = False
        return None
    
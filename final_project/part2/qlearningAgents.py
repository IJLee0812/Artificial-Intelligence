# qlearningAgents.py
# ------------------
'''
Copyright (C) Computer Science & Engineering, Soongsil University. This material is for educational uses only. 
Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. 
Written by Haneul Pyeon, November 2024.
'''


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - update
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter() # Initialize Q-values as a counter (dictionary with default value 0)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)] # Return the Q-value for the given state-action pair

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state) # Get all legal actions for the current state
        if len(legalActions) == 0:
          return 0.0
        
        tmp = util.Counter()
        
        for action in legalActions:
          tmp[action] = self.getQValue(state, action)
        
        return tmp[tmp.argMax()] 
    
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state) # Get all legal actions for the current state
        best_action = None
        max_val = float('-inf')
        
        # Iterate through actions to find the one with the highest Q-value
        for action in actions:
          q_value = self.q_values[(state, action)]
          if max_val < q_value:
            max_val = q_value
            best_action = action

        # Return the action with the highest Q-value
        return best_action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        old_q_value = self.getQValue(state, action) # Get the old Q-value for the current state-action pair
        old_part = (1 - self.alpha) * old_q_value  # Compute the non-learning portion of the Q-value update
        reward_part = self.alpha * reward # Compute the reward contribution to the Q-value
        
        if not nextState:
          self.q_values[(state, action)] = old_part + reward_part # If there is no next state, update Q-value using only the reward
        else:
          nextState_part = self.alpha * self.discount * self.getValue(nextState) # Compute the next state contribution to the Q-value update
          self.q_values[(state, action)] = old_part + reward_part + nextState_part # Update the Q-value using all components

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

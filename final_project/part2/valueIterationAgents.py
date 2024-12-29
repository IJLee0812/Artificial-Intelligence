# valueIterationAgents.py
# -----------------------
'''
Copyright (C) Computer Science & Engineering, Soongsil University. This material is for educational uses only. 
Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. 
Written by Haneul Pyeon, November 2024.
'''


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
          states = self.mdp.getStates() # Get all states in the MDP
          temp_counter = util.Counter() # Temporary counter to store new values

          for state in states:
            max_val = float("-inf")

            for action in self.mdp.getPossibleActions(state):
              q_value = self.computeQValueFromValues(state, action)  # Compute Q-value for the action

              if q_value > max_val:
                max_val = q_value # Update the maximum value if the Q-value is greater

              
              temp_counter[state] = max_val # Store the max value for the state in the temporary counter
          
          self.values = temp_counter # Update the values with the new computed values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        action_prob_pairs = self.mdp.getTransitionStatesAndProbs(state, action)
        total = 0 # Initialize the total Q-value

        for next_state, prob in action_prob_pairs:
            reward = self.mdp.getReward(state, action, next_state) # Get reward for the transition
            total += prob * (reward + self.discount * self.values[next_state]) # Compute contribution of the transition to the Q-value

        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None 
        max_val = float("-inf")

        for action in self.mdp.getPossibleActions(state):
          q_value = self.computeQValueFromValues(state, action) # Compute Q-value for the action

          if q_value > max_val:
            max_val = q_value # Update the maximum value if the Q-value is greater
            best_action = action # Update the best action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates() # Get all states in the MDP
        num_states = len(states) # Get the number of states

        for i in range(self.iterations):
          state = states[i % num_states] # Select the current state in cyclic order

          if not self.mdp.isTerminal(state): # Skip terminal states
            values = [] # Initialize a list to store Q-values for all actions

            for action in self.mdp.getPossibleActions(state):
              q_value = self.computeQValueFromValues(state, action) # Compute Q-value for the action
              values.append(q_value) # Add the Q-value to the list
            
            self.values[state] = max(values) # Update the value of the state with the maximum Q-value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq = util.PriorityQueue() # Initialize a priority queue
        predecessors = {} # Dictionary to store predecessors of each state

        for state in self.mdp.getStates():
          if not self.mdp.isTerminal(state): # Skip terminal states
            for action in self.mdp.getPossibleActions(state):
              for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                if nextState in predecessors:
                  predecessors[nextState].add(state) # Add state to predecessors set
                else:
                  predecessors[nextState] = {state} # Create a new predecessors set for the state

        for state in self.mdp.getStates():
          if not self.mdp.isTerminal(state): # Skip terminal states
            values = [] # Initialize a list to store Q-values for all actions

            for action in self.mdp.getPossibleActions(state):
              q_value = self.computeQValueFromValues(state, action) # Compute Q-value for the action
              values.append(q_value) # Add the Q-value to the list
            
            diff = abs(max(values) - self.values[state]) # Compute the difference between max Q-value and current value
            pq.update(state, - diff) # Push the state into the priority queue with negative priority

        for i in range(self.iterations):
          if pq.isEmpty():
            break
          
          temp_state = pq.pop() # Pop the state with the highest priority
          if not self.mdp.isTerminal(temp_state): # Skip terminal states
            values = [] # Initialize a list to store Q-values for all actions
            for action in self.mdp.getPossibleActions(temp_state):
              q_value = self.computeQValueFromValues(temp_state, action) # Compute Q-value for the action
              values.append(q_value) # Add the Q-value to the list
            
            self.values[temp_state] = max(values) # Update the value of the state with the maximum Q-value

          for p in predecessors[temp_state]: # Update the priority of predecessor states
            if not self.mdp.isTerminal(p): # Skip terminal states
              values = [] # Initialize a list to store Q-values for all actions
              for action in self.mdp.getPossibleActions(p):
                q_value = self.computeQValueFromValues(p, action) # Compute Q-value for the action
                values.append(q_value) # Add the Q-value to the list

              diff = abs(max(values) - self.values[p]) # Compute the difference between max Q-value and current value
              if diff > self.theta: 
                pq.update(p, -diff) # Push the state into the priority queue if the difference exceeds the threshold
# multiAgents.py
# --------------
'''
Copyright (C) Computer Science & Engineering, Soongsil University. This material is for educational uses only. 
Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. 
Written by Haneul Pyeon, October 2024.
'''


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Calculate the minimum distance to the nearest food pellet
        # The smaller this distance, the better, as Pacman aims to consume food
        newFoodList = newFood.asList()
        min_food_distance = -1
        for food in newFoodList:
            distance = util.manhattanDistance(newPos, food)
            if min_food_distance >= distance or min_food_distance == -1:
                min_food_distance = distance

        # Calculate the total distances from Pacman to all ghosts
        # Additionally, check if any ghost is too close (distance <= 1), which increases the danger to Pacman
        distances_to_ghosts = 1  # Initialize the ghost distance sum
        proximity_to_ghosts = 0 # Count how many ghosts are within close proximity
        for ghost_state in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost_state)
            distances_to_ghosts += distance  # Accumulate distances to ghosts
            if distance <= 1:
                proximity_to_ghosts += 1 # Increment the count of nearby ghosts

        # Combine the metrics into an overall evaluation score:
        # - Favor states closer to food
        # - Penalize states closer to ghosts
        # - Penalize states with nearby ghosts more heavily        
        return successorGameState.getScore() + (1 / float(min_food_distance)) - (1 / float(distances_to_ghosts)) - proximity_to_ghosts

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Define the recursize minimax function
        def miniMax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth: # Return the reward in case the defined depth is reached or the game is won/lost
                return self.evaluationFunction(gameState)
            if agent == 0: # Pacman : maximize the score
                return max(miniMax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else: # Ghosts : minimize the score
                nextAgent = agent + 1 # Calculate the next agent and increase depth accordingly
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return min(
                    miniMax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent)
                    )

        # Perform minimax on the root node (Pacman's turn)
        maxUtility = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            utility = miniMax(1, 0, gameState.generateSuccessor(0, action))
            if utility > maxUtility:
                maxUtility = utility
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Define a maximizer function for Pacman's turn
        def maximizer(agent, depth, game_state, a, b): 
            v = float("-inf") # Initalize the utility value to negative infinity
            for newState in game_state.getLegalActions(agent): # Iterate through all possible actions
                v = max(v, alpha_beta_prune(1, depth, game_state.generateSuccessor(agent, newState), a, b))
                
                # If the current utility exceeds beta, prune further exploration
                if v > b:
                    return v
                
                # Update alpha with the current best utility
                a = max(a, v)
            
            return v

        # Define a minimizer function for the Ghost's turn
        def minimizer(agent, depth, game_state, a, b): 
            v = float("inf") # Initialize the utility value to positive infinity

            # Calculate the next agent index and handle depth increment if the last agent's turn ends
            next_agent = agent + 1 # Calculate the next agent and increase depth accordingly
            if game_state.getNumAgents() == next_agent: # Wrap around to Pacman after the last ghost
                next_agent = 0
            if next_agent == 0:
                depth += 1 # Increase the depth when transitioning back to Pacman

            for newState in game_state.getLegalActions(agent): # Iterate through all possible actions
                v = min(v, alpha_beta_prune(next_agent, depth, game_state.generateSuccessor(agent, newState), a, b))

                # If the current utility is less than alpha, prune further exploration
                if v < a:
                    return v
                
                # Update beta with the current best utility
                b = min(b, v)
            
            return v
        
        # Define a alpha-beta pruning function
        def alpha_beta_prune(agent, depth, game_state, a, b):
            # Terminate the search if a win/lose state is reached or if the depth limit is met
            if game_state.isLose() or game_state.isWin() or depth == self.depth: # return the reward in case the defined depth is reached or the game is won/lost
                return self.evaluationFunction(game_state)
            
            # Call the appropriate function based on the agent type (Pacman or Ghost)
            if agent == 0: # Pacman's turn (maximize)
                return maximizer(agent, depth, game_state, a, b)
            else: # Ghost's turn (minimize)
                return minimizer(agent, depth, game_state, a, b)
            
        # Perform the maximizer function on the root node (Pacman's turn)
        utility = float("-inf")  # Initialize the utility value to negative infinity
        action = Directions.WEST  # Default action
        alpha = float("-inf")  # Initialize alpha to negative infinity
        beta = float("inf")  # Initialize beta to positive infinity

        # Iterate through Pacman's legal actions to find the best one
        for agentState in gameState.getLegalActions(0):
            ghostValue = alpha_beta_prune(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
            
            # Update the best utility and corresponding action
            if ghostValue > utility:
                utility = ghostValue
                action = agentState

             # If the utility exceeds beta, prune further exploration    
            if utility > beta:
                return utility
            
            # Update alpha with the current best utility
            alpha = max(alpha, utility)

        # Finally, return an action
        return action
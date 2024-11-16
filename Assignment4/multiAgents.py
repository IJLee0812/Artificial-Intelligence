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
        # Calculating distance to the farthest food pellet
        newFoodList = newFood.asList()
        min_food_distance = -1
        for food in newFoodList:
            distance = util.manhattanDistance(newPos, food)
            if min_food_distance >= distance or min_food_distance == -1:
                min_food_distance = distance

        # Calculating the distances from pacman to the ghosts. Also, checking for the proximity of the ghosts (at dist of 1) around pacman
        distances_to_ghosts = 1
        proximity_to_ghosts = 0
        for ghost_state in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost_state)
            distances_to_ghosts += distance
            if distance <= 1:
                proximity_to_ghosts += 1

        # Return the combination of the above calculated metrics
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
        # Define a miniMax function first
        def miniMax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth: # return the reward in case the defined depth is reached or the game is won/lost
                return self.evaluationFunction(gameState)
            if agent == 0: # maximize for pacman
                return max(miniMax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else: # minimize for ghosts
                nextAgent = agent + 1 # calculate the next agent and increase depth accordingly
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                return min(miniMax(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))

        # Performing maximize action for the root node, i.e. pacman
        maximum = float("-inf")
        action = Directions.WEST
        for agentState in gameState.getLegalActions(0):
            utility = miniMax(1, 0, gameState.generateSuccessor(0, agentState))
            if (utility > maximum or maximum == float("-inf")):
                maximum = utility
                action = agentState

        # Finally, return an action
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Define a maximizer function
        def maximizer(agent, depth, game_state, a, b): 
            v = float("-inf")
            for newState in game_state.getLegalActions(agent):
                v = max(v, alpha_beta_prune(1, depth, game_state.generateSuccessor(agent, newState), a, b))
                
                if v > b:
                    return v
                
                a = max(a, v)
            
            return v

        # Define a minimizer function
        def minimizer(agent, depth, game_state, a, b): 
            v = float("inf")

            next_agent = agent + 1 # calculate the next agent and increase depth accordingly
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
            if next_agent == 0:
                depth += 1

            for newState in game_state.getLegalActions(agent):
                v = min(v, alpha_beta_prune(next_agent, depth, game_state.generateSuccessor(agent, newState), a, b))

                if v < a:
                    return v
                
                b = min(b, v)
            
            return v
        
        # Define a Alpha-Beta pruning function
        def alpha_beta_prune(agent, depth, game_state, a, b):
            if game_state.isLose() or game_state.isWin() or depth == self.depth: # return the reward in case the defined depth is reached or the game is won/lost
                return self.evaluationFunction(game_state)
            
            if agent == 0: # maximize for pacman
                return maximizer(agent, depth, game_state, a, b)
            else: # minimize for ghosts
                return minimizer(agent, depth, game_state, a, b)
            
        # Performing maximizer function to the root node, i.e. pacman using Alpha-Beta pruning function
        utility = float("-inf")
        action = Directions.WEST
        alpha = float("-inf")
        beta = float("inf")

        for agentState in gameState.getLegalActions(0):
            ghostValue = alpha_beta_prune(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
            
            if ghostValue > utility:
                utility = ghostValue
                action = agentState
            if utility > beta:
                return utility
            
            alpha = max(alpha, utility)

        # Finally, return an action
        return action
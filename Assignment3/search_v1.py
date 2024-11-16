# search.py
# ---------
'''
Copyright (C) Computer Science & Engineering, Soongsil University. This material is for educational uses only. 
Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. 
Written by Haneul Pyeon, October 2024.
'''


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""


import util
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """


    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """


    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()  # Initialize the stack using the Stack class from util.py
    visited = set()  # Keep track of visited nodes to ensure graph search and avoid revisiting

    start_state = problem.getStartState()  # Get the initial state of the problem
    stack.push((start_state, []))  # Push the initial state and an empty action list onto the stack

    while not stack.isEmpty():
        current_state, actions = stack.pop()  # Pop the current state and actions from the stack

        if problem.isGoalState(current_state):  # Check if the current state is the goal state
            return actions  # If it is the goal state, return the list of actions

        if current_state not in visited:  # Check if the current state has been visited
            visited.add(current_state)  # If not, mark it as visited
            successors = problem.getSuccessors(current_state)  # Get the successors of the current state

            for next_state, action, cost in successors:
                if next_state not in visited:  # Check if the successor state has been visited
                    new_actions = actions + [action]  # Add the action to the list of actions
                    stack.push((next_state, new_actions))  # Push the successor state and actions onto the stack

    return []  # If the goal state is not reached, return an empty list


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()  # Initialize the queue using the Queue class from util.py
    visited = set()  # Keep track of visited nodes to ensure graph search and avoid revisiting

    start_state = problem.getStartState()  # Get the initial state of the problem
    queue.push((start_state, []))  # Push the initial state and an empty action list onto the queue

    while not queue.isEmpty():
        current_state, actions = queue.pop()  # Pop the current state and actions from the queue

        if problem.isGoalState(current_state):  # Check if the current state is the goal state
            return actions  # If it is the goal state, return the list of actions

        if current_state not in visited:  # Check if the current state has been visited
            visited.add(current_state)  # If not, mark it as visited
            successors = problem.getSuccessors(current_state)  # Get the successors of the current state

            for next_state, action, cost in successors:
                if next_state not in visited:  # Check if the successor state has been visited
                    new_actions = actions + [action]  # Add the action to the list of actions
                    queue.push((next_state, new_actions))  # Push the successor state and actions onto the queue

    return []  # If the goal state is not reached, return an empty list


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    p_queue = util.PriorityQueue()  # Initialize the priority queue using the PriorityQueue class from util.py
    visited = set()  # Keep track of visited nodes to ensure graph search and avoid revisiting

    start_state = problem.getStartState()  # Get the initial state of the problem
    p_queue.push((start_state, []), 0)  # Push the initial state, an empty action list, and priority 0 onto the queue

    while not p_queue.isEmpty():
        current_state, actions = p_queue.pop()  # Pop the current state and actions from the queue

        if problem.isGoalState(current_state):  # Check if the current state is the goal state
            return actions  # If it is the goal state, return the list of actions

        if current_state not in visited:  # Check if the current state has been visited
            visited.add(current_state)  # If not, mark it as visited
            successors = problem.getSuccessors(current_state)  # Get the successors of the current state

            for next_state, action, cost in successors:
                if next_state not in visited:  # Check if the successor state has been visited
                    new_actions = actions + [action]  # Add the action to the list of actions
                    new_cost = problem.getCostOfActions(new_actions)  # Calculate the new cost
                    p_queue.push((next_state, new_actions), new_cost)  # Push the successor state, actions, and new cost onto the queue

    return []  # If the goal state is not reached, return an empty list


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    p_queue = util.PriorityQueue()  # Initialize the priority queue using the PriorityQueue class from util.py
    visited = set()  # Keep track of visited nodes to ensure graph search and avoid revisiting

    start_state = problem.getStartState()  # Get the initial state of the problem
    start_cost = 0  # Initialize the starting cost
    start_heuristic = heuristic(start_state, problem)  # Calculate the heuristic value for the starting state
    p_queue.push((start_state, []), start_cost + start_heuristic)  # Push the initial state, actions, and priority onto the queue

    while not p_queue.isEmpty():
        current_state, actions = p_queue.pop()  # Pop the current state and actions from the queue

        if problem.isGoalState(current_state):  # Check if the current state is the goal state
            return actions  # If it is the goal state, return the list of actions

        if current_state not in visited:  # Check if the current state has been visited
            visited.add(current_state)  # If not, mark it as visited
            successors = problem.getSuccessors(current_state)  # Get the successors of the current state

            for next_state, action, cost in successors:
                if next_state not in visited:  # Check if the successor state has been visited
                    new_actions = actions + [action]  # Add the action to the list of actions
                    new_cost = problem.getCostOfActions(new_actions)  # Calculate the new cost
                    new_heuristic = heuristic(next_state, problem)  # Calculate the heuristic value for the successor state
                    priority = new_cost + new_heuristic  # Calculate the priority as the sum of the cost and heuristic
                    p_queue.push((next_state, new_actions), priority)  # Push the successor state, actions, and priority onto the queue

    return []  # If the goal state is not reached, return an empty list


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
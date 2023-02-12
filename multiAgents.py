# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        import sys
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currentFood = currentGameState.getFood()

        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            if manhattanDistance(newPos, ghostState.getPosition()) < 7 and scaredTime <= 2:
                return -100
        
        pq = util.PriorityQueue()
        distances = [manhattanDistance(newPos, foodPos) for foodPos in currentFood.asList()]
        visitedStates = set()
        if not distances:
            min_distance = 0
        else:
            min_distance = min(distances)
        pq.push((successorGameState, action, 0), min_distance)

        while not pq.isEmpty():
            state, action, cost = pq.pop()
            pos = state.getPacmanPosition()
            distances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
            if not distances:
                min_distance = 0
            else:
                min_distance = min(distances)

            if pos in currentFood.asList():
                return state.getScore()

            if pos not in visitedStates:
                visitedStates.add(pos)
                for new_action in state.getLegalActions():
                    new_state = state.generatePacmanSuccessor(new_action)
                    pq.push((new_state, new_action, cost + 1), min_distance + cost)

        return -1
            

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        import sys
        actions = gameState.getLegalActions(0)
        bestActionScore = -sys.maxsize

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            actionScore = self.get_value(successor, 1, 0)

            if actionScore > bestActionScore:
                bestAction = action
                bestActionScore = actionScore

        return bestAction


    def get_value(self, gameState, agentIndex, currentDepth):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.max_value(gameState, currentDepth)
        else:
            return self.min_value(gameState, agentIndex, currentDepth)

    
    def max_value(self, gameState: GameState, currentDepth):
        import sys
        v = -sys.maxsize
        actions = gameState.getLegalActions(0)
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            v = max(v, self.get_value(successor, 1, currentDepth))

        return v
    

    def min_value(self, gameState: GameState, agentIndex, currentDepth):
        import sys
        v = sys.maxsize
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            successorIndex = 0
            currentDepth += 1
        else:
            successorIndex = agentIndex + 1

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v = min(v, self.get_value(successor, successorIndex, currentDepth))

        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import sys
        actions = gameState.getLegalActions(0)
        bestActionScore = -sys.maxsize
        alpha = -sys.maxsize
        beta = sys.maxsize
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            actionScore = self.get_value(successor, 1, 0, alpha, beta)

            if actionScore > bestActionScore:
                bestAction = action
                bestActionScore = actionScore
            
            alpha = max(alpha, bestActionScore)

        return bestAction


    def get_value(self, gameState, agentIndex, currentDepth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.max_value(gameState, currentDepth, alpha, beta)
        else:
            return self.min_value(gameState, agentIndex, currentDepth, alpha, beta)

    
    def max_value(self, gameState: GameState, currentDepth, alpha, beta):
        import sys
        v = -sys.maxsize
        actions = gameState.getLegalActions(0)
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            v = max(v, self.get_value(successor, 1, currentDepth, alpha, beta))
            if v > beta:
                return v 
            alpha = max(alpha, v)

        return v
    

    def min_value(self, gameState: GameState, agentIndex, currentDepth, alpha, beta):
        import sys
        v = sys.maxsize
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            successorIndex = 0
            currentDepth += 1
        else:
            successorIndex = agentIndex + 1

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v = min(v, self.get_value(successor, successorIndex, currentDepth, alpha, beta))
            if v < alpha:
                return v 
            beta = min(beta, v)

        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        import sys
        actions = gameState.getLegalActions(0)
        bestActionScore = -sys.maxsize

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            actionScore = self.get_value(successor, 1, 0)

            if actionScore > bestActionScore:
                bestAction = action
                bestActionScore = actionScore

        return bestAction


    def get_value(self, gameState, agentIndex, currentDepth):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.max_value(gameState, currentDepth)
        else:
            return self.exp_value(gameState, agentIndex, currentDepth)

    
    def max_value(self, gameState: GameState, currentDepth):
        import sys
        v = -sys.maxsize
        actions = gameState.getLegalActions(0)
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            v = max(v, self.get_value(successor, 1, currentDepth))

        return v
    

    def exp_value(self, gameState: GameState, agentIndex, currentDepth):
        import sys
        v = 0
        actions = gameState.getLegalActions(agentIndex)
        if agentIndex == gameState.getNumAgents() - 1:
            successorIndex = 0
            currentDepth += 1
        else:
            successorIndex = agentIndex + 1

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v += self.get_value(successor, successorIndex, currentDepth) / len(actions)

        return v

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    import sys

    pacmanPos = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    ghostsPos = currentGameState.getGhostPositions()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    "*** YOUR CODE HERE ***"
    for ghostPos, scaredTime in zip(ghostsPos, scaredTimes):
        if (pacmanPos[0] == ghostPos[0] and pacmanPos[1] == ghostPos[1]) and scaredTime <= 2:
            return -100
    if pacmanPos in foodPositions or pacmanPos in currentGameState.getCapsules():
        return currentGameState.getScore() + 100
    if not foodPositions:
        return currentGameState.getScore() + 100
    min_food_dist = min(manhattanDistance(pacmanPos, foodPos) for foodPos in foodPositions)
    min_ghost_dist = min(manhattanDistance(pacmanPos, ghostState.getPosition()) * (1-ghostState.scaredTimer) for ghostState in ghostStates)


    return currentGameState.getScore() + 1/min_food_dist - 1/min_ghost_dist

# Abbreviation
better = betterEvaluationFunction

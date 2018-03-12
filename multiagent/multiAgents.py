# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import sys

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        '''
        print successorGameState
        print newPos
        print newFood
        print newGhostStates
        print newScaredTimes
        '''

        ghostPositions = successorGameState.getGhostPositions()
        for gpostion in ghostPositions:
             if newPos == gpostion:
                 return -sys.maxint - 1

        currentFood = currentGameState.getFood()
        if currentFood[newPos[0]][newPos[1]]:
            return sys.maxint

        closestX = 0
        closestY = 0
        dis = sys.maxint
        foodsList = newFood.asList()
        for food in foodsList:
            d = abs(newPos[0] - food[0]) + abs(newPos[1] - food[1])
            if d < dis:
                dis = d
                closestX = food[0]
                closestY = food[1]
        currentPos = currentGameState.getPacmanPosition()
        originalDis = abs(currentPos[0] - closestX) + abs(currentPos[1] - closestY)
        return originalDis - dis

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
        """
        "*** YOUR CODE HERE ***"
        def value(state, depth, agentIndex, actions):
            if agentIndex >= state.getNumAgents():
                agentIndex = 0
                depth += 1
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return pacmanValue(state, depth, agentIndex, actions)
            else:
                return ghostValue(state, depth, agentIndex, actions)

        def pacmanValue(state, depth, agentIndex, actions):
            max = -sys.maxint - 1
            pacActions = state.getLegalActions(agentIndex)

            if not pacActions:
                return self.evaluationFunction(state)

            resultAction = pacActions[0]
            for action in pacActions:
                successor = state.generateSuccessor(agentIndex, action)
                val = value(successor, depth, agentIndex + 1, actions)
                if val > max:
                    max = val
                    resultAction = action
            actions.append(resultAction)
            return max

        def ghostValue(state, depth, agentIndex, actions):
            min = sys.maxint
            ghostActions = state.getLegalActions(agentIndex)

            if not ghostActions:
                return self.evaluationFunction(state)

            for action in ghostActions:
                successor = state.generateSuccessor(agentIndex, action)
                val = value(successor, depth, agentIndex + 1, actions)
                if val < min:
                    min = val
            return min

        actions = []
        value(gameState, 0, 0, actions)
        return actions[len(actions) - 1]
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBetaValue(state, depth, agentIndex, actions, alpha, beta):
            if agentIndex >= state.getNumAgents():
                agentIndex = 0
                depth += 1
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return pacmanValue(state, depth, agentIndex, actions, alpha, beta)
            else:
                return ghostValue(state, depth, agentIndex, actions, alpha, beta)

        def pacmanValue(state, depth, agentIndex, actions, alpha, beta):
            v = -sys.maxint - 1
            pacActions = state.getLegalActions(agentIndex)

            if not pacActions:
                return self.evaluationFunction(state)

            resultAction = pacActions[0]
            for action in pacActions:
                successor = state.generateSuccessor(agentIndex, action)
                val = alphaBetaValue(successor, depth, agentIndex + 1, actions, alpha, beta)
                if val > v:
                    v = val
                    resultAction = action
                if val > beta:
                    actions.append(action)
                    return v
                alpha = max(alpha, v)
            actions.append(resultAction)
            return v

        def ghostValue(state, depth, agentIndex, actions, alpha, beta):
            v = sys.maxint
            ghostActions = state.getLegalActions(agentIndex)

            if not ghostActions:
                return self.evaluationFunction(state)

            for action in ghostActions:
                successor = state.generateSuccessor(agentIndex, action)
                val = alphaBetaValue(successor, depth, agentIndex + 1, actions, alpha, beta)
                if val < v:
                    v = val
                if val < alpha:
                    return v
                beta = min(beta, v)
            return v

        actions = []
        alphaBetaValue(gameState, 0, 0, actions, -sys.maxint - 1, sys.maxint)
        return actions[len(actions) - 1]
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(state, depth, agentIndex, actions):
            if agentIndex >= state.getNumAgents():
                agentIndex = 0
                depth += 1
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return pacmanValue(state, depth, agentIndex, actions)
            else:
                return expValue(state, depth, agentIndex, actions)

        def pacmanValue(state, depth, agentIndex, actions):
            pacActions = state.getLegalActions(agentIndex)
            values = []

            if not pacActions:
                return self.evaluationFunction(state)

            for action in pacActions:
                successor = state.generateSuccessor(agentIndex, action)
                values.append(value(successor, depth, agentIndex + 1, actions))

            maxVal = max(values)
            actions.append(pacActions[values.index(maxVal)])
            return maxVal

        def expValue(state, depth, agentIndex, actions):
            ghostActions = state.getLegalActions(agentIndex)
            values = []

            if not ghostActions:
                return self.evaluationFunction(state)

            weight = 1.0 / len(ghostActions)
            for action in ghostActions:
                successor = state.generateSuccessor(agentIndex, action)
                values.append(value(successor, depth, agentIndex + 1, actions))
            expectation = sum(values) * weight
            return expectation

        actions = []
        value(gameState, 0, 0, actions)
        return actions[len(actions) - 1]
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      I consider the following features:
      - the score
      - the minimum distance between pacman and foods
      - the total distance between pacman and all foods
      - the total distance between pacman and ghost(within 5 manhattan distance).
        because the far away ghost have less importance.
      - the scared time. If a ghost can be eaten by pacman, pacman should eat it
        to earn more points.
      - the total number of foods. The less the btter.
      - the closet capsule distance. if the distance is smaller than 3, the pacman
        should eat it.
      score + 2.0/mindis + 0.5/disFood + disGhost * 2.0 + scaredTime + 5.0/numFood + 5.0/closetCap
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return sys.maxint
    if currentGameState.isLose():
        return -sys.maxint - 1

    pacmanPosition = currentGameState.getPacmanPosition()
    allfoods = (currentGameState.getFood()).asList()
    allghosts = currentGameState.getGhostPositions()
    actions = currentGameState.getLegalActions()
    score = currentGameState.getScore()
    capPos = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    numFood = len(allfoods)
    disFood = 0
    mindis = sys.maxint
    for food in allfoods:
        disFood += manhattanDistance(pacmanPosition, food)
        if manhattanDistance(pacmanPosition, food) < mindis:
            mindis = manhattanDistance(pacmanPosition, food)

    disGhost = 0
    scaredTime = 0
    for ghost in allghosts:
        dis = manhattanDistance(pacmanPosition, ghost)
        if dis < 5:
            disGhost += dis
    if sum(scaredTimes) != 0:
        disGhost = 1 / sum(manhattanDistance(pacmanPosition, ghost) for ghost in allghosts)
        scaredTime = sum(scaredTimes)

    closetCap = 1
    if capPos:
        closetCap = min(manhattanDistance(cap, pacmanPosition) for cap in capPos)
        if closetCap <= 3:
            closetCap /= 4.0

    return score + 2.0/mindis + 0.5/disFood + disGhost * 2.0 + scaredTime + 5.0/numFood + 5.0/closetCap
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

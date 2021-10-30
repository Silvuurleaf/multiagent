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
        scores = [self.evaluationFunction(gameState, action) for action in
                  legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if
                       scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state,
        like the remaining food (newFood)
        Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
            scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in
                          newGhostStates]

        print("newGhostStates: {}".format(newGhostStates[0]))
        print("scarded clock: {}".format(newScaredTimes))

        totalFood = currentGameState.getFood().count()

        # food locations
        newFood = successorGameState.getFood()
        # pacmans position
        pacPosition = successorGameState.getPacmanPosition()

        farthest = 0
        # food that's farthest away
        for food in newFood.asList():
            distanceToFood = manhattanDistance(pacPosition, food)
            if farthest < distanceToFood:
                farthest = distanceToFood
        # FIXME need a way to make pacman not scared
        # add bonus if the ghost is scared

        evaluation_value = farthest

        for ghost in newGhostStates:
            ghost_position = ghost.getPosition()
            distanceToGhost = manhattanDistance(pacPosition, ghost_position)
            # something increase exponentially as the ghost gets closer
            evaluation_value += 4 ** (
                        2 - (1 / distanceToGhost))  # closer -> larger exponent
        return evaluation_value

        # return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        self.num_agents = gameState.getNumAgents()
        self.ghosts = []
        for i in range(self.num_agents - 1):
            self.ghosts.append(i + 1)

        depth = 0
        best_action = 0
        best_value = float('-inf')

        # all the legal actions pacman can currently take
        pacman_actions = gameState.getLegalActions(0)
        for action in pacman_actions:
            pacman_future = gameState.generateSuccessor(0, action)
            value = self.Minimize_Action(pacman_future, self.ghosts[0], depth)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def Maximize_Action(self, gameState, depth):
        pacman_actions = gameState.getLegalActions(0)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        if not pacman_actions:
            return self.evaluationFunction(gameState)

        values = []
        for action in pacman_actions:
            pacman_future = gameState.generateSuccessor(0, action)
            values.append(
                self.Minimize_Action(pacman_future, self.ghosts[0], depth))

        return max(values)

    def Minimize_Action(self, gameState, ghost_ID, depth):

        """ For the MIN Players or Agents  """

        # gets the legal actions for ghost
        ghost_actions = gameState.getLegalActions(ghost_ID)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        if not ghost_actions:
            return self.evaluationFunction(gameState)

        min_reward = float('inf')
        values = []

        if ghost_ID < gameState.getNumAgents() - 1:
            for action in ghost_actions:
                # future caused by ghost taking its action
                ghost_future = gameState.generateSuccessor(ghost_ID, action)
                # every action of ghost X needs to be evaluated by ghost X+1
                # until run out of ghosts
                values.append(
                    self.Minimize_Action(ghost_future, ghost_ID + 1, depth))
        else:
            for action in ghost_actions:
                ghost_future = gameState.generateSuccessor(ghost_ID, action)
                values.append(self.Maximize_Action(ghost_future, depth + 1))

        if not values:
            return min_reward
        else:
            return min(values)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        beta = float('inf')
        alpha = float('-inf')

        self.num_agents = gameState.getNumAgents()
        self.ghosts = []
        for i in range(self.num_agents - 1):
            self.ghosts.append(i + 1)

        depth = 0
        best_action = 0
        best_value = float('-inf')

        # all the legal actions pacman can currently take
        pacman_actions = gameState.getLegalActions(0)
        for action in pacman_actions:
            pacman_future = gameState.generateSuccessor(0, action)
            value = self.Minimize_Action(pacman_future, self.ghosts[0], depth, alpha,
                                   beta)

            if alpha > beta:
                return action

            if value > alpha:
                alpha = value
                best_action = action

        return best_action

    def Maximize_Action(self, gameState, depth, alpha, beta):
        pacman_actions = gameState.getLegalActions(0)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        if not pacman_actions:
            return self.evaluationFunction(gameState)

        value = float('-inf')
        for action in pacman_actions:
            pacman_future = gameState.generateSuccessor(0, action)
            value = max(
                self.Minimize_Action(pacman_future, self.ghosts[0], depth, alpha,
                               beta), value)

            alpha = max(alpha, value)

            if alpha > beta:
                return value

        # return the value not alpha, because return value is only alpha, if
        # alpha was changed in this tree and returned as the value. Alpha
        # may also be a value carried over from another subtree.

        return value

    def Minimize_Action(self, gameState, ghost_ID, depth, alpha, beta):
    
        # gets the legal actions for ghost
        ghost_actions = gameState.getLegalActions(ghost_ID)

        if depth == self.depth:
            return self.evaluationFunction(gameState)

        if not ghost_actions:
            return self.evaluationFunction(gameState)


        value = float('inf')
        if ghost_ID < gameState.getNumAgents()-1:
            for action in ghost_actions:
                # future caused by ghost taking its action
                ghost_future = gameState.generateSuccessor(ghost_ID, action)
                # every action of ghost X needs to be evaluated by ghost X+1
                # until run out of ghosts
                value = min(self.Minimize_Action(ghost_future, ghost_ID + 1, depth, alpha, beta), value)

                if value < alpha:
                    return value

                beta = min(beta, value)
        else:
            for action in ghost_actions:
                ghost_future = gameState.generateSuccessor(ghost_ID, action)
                value = min(self.Maximize_Action(ghost_future, depth + 1, alpha, beta), value)

                if value < alpha:
                    return value

                beta = min(beta, value)

        return value


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
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction

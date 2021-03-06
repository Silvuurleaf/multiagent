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
        ghost_states = successorGameState.getGhostStates()
        food_list = successorGameState.getFood().asList()
        currentFood = currentGameState.getFood()
        pacman_position = successorGameState.getPacmanPosition()

        score = 0

        # Check if the next state is winning/losing
        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')

        # checks to see if pacman ate some food
        pacX, pacY = pacman_position
        if currentFood[pacX][pacY]:
            #print(currentFood[pacX][pacY])
            score += 1

        food_distances = {}
        if food_list:
            for food in food_list:
                food_distances[food] = manhattanDistance(food, pacman_position)

            closest_food = min(food_distances, key=food_distances.get)
            for ghost in ghost_states:
                if ghost.scaredTimer > 0:
                    score += 2 / food_distances[closest_food]
                    food_list.remove(closest_food)
                else:
                    score += 1 / food_distances[closest_food]
                    food_list.remove(closest_food)

        # distance from ghost
        if ghost_states:
            for ghost in ghost_states:
                ghost_distance = manhattanDistance(pacman_position,
                                                   ghost.getPosition())

                # if ghost is close return large negative value
                if ghost_distance <= 1:
                    return float('-inf')

                # otherwise subtract the inverse (closer -> larger negative)
                score -= 1 / ghost_distance

        return score


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

        # Total number of agents (including pacman)
        self.num_agents = gameState.getNumAgents()

        # create a list with the ID of all of the ghosts
        self.ghosts = []
        for i in range(self.num_agents - 1):
            self.ghosts.append(i + 1)

        depth = 0                   # current depth
        best_action = None          # best course of action
        best_value = float('-inf')  # score for an evaluated action

        # all the legal actions pacman can currently take
        pacman_actions = gameState.getLegalActions(0)

        # Iterate through all of possible actions
        for action in pacman_actions:
            # future game state after pacman action taken
            pacman_future = gameState.generateSuccessor(0, action)
            # return from applying min node for a ghost agent on pacman actions
            value = self.Minimize_Action(pacman_future, self.ghosts[0], depth)

            # Pacman takes the best value, cause its a max node
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def Maximize_Action(self, gameState, depth):
        """
            Apply max node implementation
        """

        # Legal pacman actions
        pacman_actions = gameState.getLegalActions(0)

        # see if we have reached the depth limit, return value of state
        if depth == self.depth:
            return self.evaluationFunction(gameState)

        # if there are not actions left return value of state
        if not pacman_actions:
            return self.evaluationFunction(gameState)

        # store values for each node for a given pacman action
        values = []
        for action in pacman_actions:
            pacman_future = gameState.generateSuccessor(0, action)
            # for a given gamestate min value for each
            # state resulting from said action
            values.append(
                self.Minimize_Action(pacman_future, self.ghosts[0], depth))

        # for all the nodes return the largest value
        return max(values)

    def Minimize_Action(self, gameState, ghost_ID, depth):

        """ For the MIN Players or Agents  """

        # gets the legal actions for ghost
        ghost_actions = gameState.getLegalActions(ghost_ID)

        # see if we have reached the depth limit, return value of state
        if depth == self.depth:
            return self.evaluationFunction(gameState)

        # if there are not actions left return value of state
        if not ghost_actions:
            return self.evaluationFunction(gameState)

        min_reward = float('inf')
        values = []

        # check we aren't the last ghost
        if ghost_ID < gameState.getNumAgents() - 1:
            for action in ghost_actions:
                # future caused by ghost taking its action
                ghost_future = gameState.generateSuccessor(ghost_ID, action)
                # every action of ghost X needs to be evaluated by ghost X+1
                # until run out of ghosts
                values.append(
                    self.Minimize_Action(ghost_future, ghost_ID + 1, depth))
        else:
            # last ghost need to call maximize afterwards
            for action in ghost_actions:
                ghost_future = gameState.generateSuccessor(ghost_ID, action)
                values.append(self.Maximize_Action(ghost_future, depth + 1))

        # make sure not empty list we are calling min on
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

        # default values for alpha/beta pruning
        beta = float('inf')
        alpha = float('-inf')

        self.num_agents = gameState.getNumAgents()
        self.ghosts = []
        for i in range(self.num_agents - 1):
            self.ghosts.append(i + 1)

        depth = 0
        best_action = 0

        # all the legal actions pacman can currently take
        pacman_actions = gameState.getLegalActions(0)
        for action in pacman_actions:
            # create future state
            pacman_future = gameState.generateSuccessor(0, action)
            # call minimize
            value = self.Minimize_Action(pacman_future, self.ghosts[0], depth,
                                         alpha,
                                         beta)

            # if alpha > beta -> prune, just return current action
            if alpha > beta:
                return action

            # otherwise acts like a max node, reassign alpha
            if value > alpha:
                alpha = value
                best_action = action

        return best_action

    def Maximize_Action(self, gameState, depth, alpha, beta):
        """
            Max node for alpha beta pruning
        """
        pacman_actions = gameState.getLegalActions(0)

        # see if we have reached the depth limit, return value of state
        if depth == self.depth:
            return self.evaluationFunction(gameState)

        # if no actions left return currents tate value
        if not pacman_actions:
            return self.evaluationFunction(gameState)

        value = float('-inf')
        # for every action call min node on it, pass alpha, beta values
        for action in pacman_actions:
            pacman_future = gameState.generateSuccessor(0, action)
            value = max(
                self.Minimize_Action(pacman_future, self.ghosts[0], depth,
                                     alpha,
                                     beta), value)

            # max node need to choose the largest value for alpha
            alpha = max(alpha, value)

            # if alpha > beta -> prune
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
        if ghost_ID < gameState.getNumAgents() - 1:
            for action in ghost_actions:
                # future caused by ghost taking its action
                ghost_future = gameState.generateSuccessor(ghost_ID, action)
                # every action of ghost X needs to be evaluated by ghost X+1
                # until run out of ghosts
                value = min(
                    self.Minimize_Action(ghost_future, ghost_ID + 1, depth,
                                         alpha, beta), value)

                if value < alpha:
                    return value

                beta = min(beta, value)
        else:
            for action in ghost_actions:
                ghost_future = gameState.generateSuccessor(ghost_ID, action)
                value = min(
                    self.Maximize_Action(ghost_future, depth + 1, alpha, beta),
                    value)

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

        # pacman should still behave optimally and not based on chance
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
            return 0
        else:
            # since expectimax is uniform we can take the sum and divide
            # by the number of actions
            return sum(values) / len(ghost_actions)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    score = currentGameState.getScore()
    pacman_position = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    currentFood = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    capsules_list = currentGameState.getCapsules()


    # Checks if the gameState will result in losing
    if currentGameState.isLose():
        return float('-inf')

    # Checks if the gameState will result in wining
    if currentGameState.isWin():
        return float('inf')

    """
        Not sure why this code isn't working
        Is able to pass without it though.
        
        Was a modifier to see if pacman ate food at 
        his given location
        
    pacX, pacY = pacman_position
    if currentFood[pacX][pacY]:
        score += 1
    """

    # distance from ghosts
    for ghost in ghost_states:
        ghost_distance = manhattanDistance(pacman_position, ghost.getPosition())
        if ghost_distance <= 1:
            return float('-inf')

        score -= 1 / ghost_distance


    # handles food distances, while there is food avaliable find the distance
    # from pacman to all of the food and find the one closest to us

    # then we check if ghosts are scared of us if they are add bonus points
    # to encourage pacman to eat food and not be scared of the ghosts
    # if they aren't scared the modifier is less.

    food_distances = {}
    if food_list:
        for food in food_list:
            food_distances[food] = manhattanDistance(food, pacman_position)

        closest_food = min(food_distances, key=food_distances.get)
        for ghost in ghost_states:
            if ghost.scaredTimer > 0:
                score += 3 / food_distances[closest_food]

                # once food is grabbed remove from the list
                food_list.remove(closest_food)
            else:
                score += 1 / food_distances[closest_food]
                # once food is grabbed remove from the list
                food_list.remove(closest_food)

    # while there are still capsules left subtract a small number
    # value was found via iteration
    if capsules_list:
        score -= .18*len(capsules_list)

    return score


# Abbreviation
better = betterEvaluationFunction

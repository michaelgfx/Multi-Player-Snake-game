from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum
from time import time


def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    # Insert your code here...

    if not state.snakes[player_index].alive:
        return state.snakes[player_index].length
    discount_factor = 0.5

    max_possible_fruits = len(state.fruits_locations) + sum([s.length for s in state.snakes
                                                             if s.index != player_index and s.alive])
    turns_left = (state.game_duration_in_turns - state.turn_number)
    max_possible_fruits = min(max_possible_fruits, turns_left)
    optimistic_future_reward = discount_factor * (1 - discount_factor ** max_possible_fruits) / (1 - discount_factor)
    original_greedy_value = state.snakes[player_index].length + optimistic_future_reward
    best_dist = state.board_size.width + state.board_size.height
    for fruit in state.fruits_locations:
        d_x = abs(state.snakes[player_index].head[0] - fruit[0])
        d_y = abs(state.snakes[player_index].head[1] - fruit[1])
        manhattan_dist = d_x + d_y
        if 0 <= manhattan_dist < best_dist:
            best_dist = manhattan_dist
    # better bonus for lower best_dist
    if best_dist != 0:
        bonus_dist_fruit = optimistic_future_reward / best_dist
    else:
        bonus_dist_fruit = optimistic_future_reward
    # better bonus if next state isn't a border
    bonus_border = 0
    head_x = state.snakes[player_index].head[0]
    head_y = state.snakes[player_index].head[1]
    radios = 1
    if not state.is_within_grid_boundaries((head_x + radios, head_y)) or not state.is_within_grid_boundaries((head_x, head_y+radios))\
            or not state.is_within_grid_boundaries((head_x - radios, head_y)) or not state.is_within_grid_boundaries((head_x, head_y-radios)):
        bonus_border -= optimistic_future_reward/(best_dist+radios)
    # if next move isn't a snake's body then bonus 0 else minus optimistic_future_reward
    bonus_is_snake_in_cell = 0
    for s in state.snakes:
        if not s.index == player_index:
            if s.is_in_cell((head_x + radios, head_y)) or s.is_in_cell((head_x, head_y + radios)) \
                    or s.is_in_cell((head_x - radios, head_y)) or s.is_in_cell((head_x, head_y - radios)):
                bonus_border -= optimistic_future_reward / (best_dist)
                break
        elif state.snakes[player_index].is_in_cell((head_x+radios, head_y)) or \
            state.snakes[player_index].is_in_cell((head_x, head_y+radios))\
            or state.snakes[player_index].is_in_cell((head_x - radios, head_y)) \
                or state.snakes[player_index].is_in_cell((head_x, head_y - radios)):
            bonus_border -= optimistic_future_reward
            break

    ret_h_val = float(original_greedy_value + bonus_dist_fruit + bonus_border + bonus_is_snake_in_cell)
    return ret_h_val


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """

    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """

        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def minimax_value(self, state: TurnBasedGameState, agent_to_play, depth):
        if state.game_state.is_terminal_state or depth == 0:
            return heuristic(state.game_state, self.player_index)
        turn = state.turn
        if turn == agent_to_play:
            cur_max = float('-inf')
            for action in state.game_state.get_possible_actions(self.player_index):
                state.agent_action = action
                v = self.minimax_value(state, agent_to_play, depth)
                cur_max = max(v, cur_max)
            return cur_max
        else:
            cur_min = float('inf')
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                turn_next_state = self.TurnBasedGameState(next_state, None)
                v = self.minimax_value(turn_next_state, agent_to_play, depth - 1)
                cur_min = min(v, cur_min)
            return cur_min

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
    
        max_actions = []
        best_value = -np.inf
        for action in state.get_possible_actions(player_index=self.player_index):
            turn_next_state = self.TurnBasedGameState(state, action)
            min_max_value = self.minimax_value(turn_next_state, MinimaxAgent.Turn.AGENT_TURN, 2)
            if min_max_value > best_value:
                best_value = min_max_value
                max_actions = [action]
            elif min_max_value == best_value:
                max_actions.append(action)


        return np.random.choice(max_actions)


class AlphaBetaAgent(MinimaxAgent):
    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...

        max_actions = []
        best_value = -np.inf
        for action in state.get_possible_actions(player_index=self.player_index):
            turn_next_state = self.TurnBasedGameState(state, action)
            min_max_value = self.alpha_beta_value(turn_next_state, MinimaxAgent.Turn.AGENT_TURN, 2, float('-inf'),
                                                  float('inf'))
            if min_max_value > best_value:
                best_value = min_max_value
                max_actions = [action]
            elif min_max_value == best_value:
                max_actions.append(action)

        return np.random.choice(max_actions)

    def alpha_beta_value(self, state: MinimaxAgent.TurnBasedGameState, agent_to_play, depth, alpha, beta):
        if state.game_state.is_terminal_state or depth == 0:
            return heuristic(state.game_state, self.player_index)
        turn = state.turn
        if turn == agent_to_play:
            cur_max = float('-inf')
            for action in state.game_state.get_possible_actions(self.player_index):
                state.agent_action = action
                v = self.alpha_beta_value(state, agent_to_play, depth, alpha, beta)
                cur_max = max(v, cur_max)
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return float('inf')
            return cur_max
        else:
            cur_min = float('inf')
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                turn_next_state = self.TurnBasedGameState(next_state, None)
                v = self.alpha_beta_value(turn_next_state, agent_to_play, depth - 1, alpha, beta)
                cur_min = min(v, cur_min)
                if cur_min <= alpha:
                    return float('-inf')
            return cur_min


def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    # 1) pick an initial state
    # np.random.choice(max_actions)
    n = 50
    random_vec = [GameAction(np.random.choice([0, 1, 2])) for _ in range(n)]
    for i in range(n):
        best_res = 0
        best_choice = []
        for new_action in [0, 1, 2]:
            random_vec[i] = GameAction(new_action)
            tuple_vec = tuple(random_vec)
            res = get_fitness(tuple_vec)
            if res > best_res:
                best_choice = [new_action]
                best_res = res
            elif res == best_res:
                best_choice.append(new_action)

        random_vec[i] = GameAction(np.random.choice(best_choice))
    print(random_vec)



def crossover(parent1, parent2, offspring_size):
    offspring_one = [GameAction(np.random.choice([0, 1, 2])) for _ in range(50)]
    offspring_two = [GameAction(np.random.choice([0, 1, 2])) for _ in range(50)]
    for k in range(int(offspring_size / 5)):
        for i in range(5):
            if k % 2 == 1:
                offspring_one[k * 5 + i] = parent1[k * 5 + i]
                offspring_two[k * 5 + i] = parent2[k * 5 + i]
            else:
                offspring_one[k * 5 + i] = parent2[k * 5 + i]
                offspring_two[k * 5 + i] = parent1[k * 5 + i]
    for i in range(5):
        offspring_one[np.random.choice(range(50))] = GameAction(np.random.choice([0, 1, 2]))
        offspring_two[np.random.choice(range(50))] = GameAction(np.random.choice([0, 1, 2]))
    res = [offspring_one, offspring_two]
    return tuple(res)


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """

    # Creating the initial population.
    new_population = []
    n = 50
    for j in range(3):
        random_vec = [GameAction(np.random.choice([0, 1, 2])) for _ in range(n)]
        for i in range(n):
            best_res = 0
            best_choice = []
            for new_action in [0, 1, 2]:
                random_vec[i] = GameAction(new_action)
                tuple_vec = tuple(random_vec)
                res = get_fitness(tuple_vec)
                if res > best_res:
                    best_choice = [new_action]
                    best_res = res
                elif res == best_res:
                    best_choice.append(new_action)

            random_vec[i] = GameAction(np.random.choice(best_choice))
            new_population.append(random_vec)

    num_generations = 24

    parent1_f = -1
    parent2_f = -1
    parent1 = []
    parent2 = []
    for generation in range(num_generations):
        for moves in new_population:
            fitness = get_fitness(tuple(moves))
            if fitness > parent1_f:
                parent1_f = fitness
                parent1 = moves
            elif fitness > parent2_f:
                parent2_f = fitness
                parent2 = moves
        temp_res = crossover(parent1, parent2, 50)
        child1 = temp_res[0]
        child2 = temp_res[1]
        new_population = [parent1, parent2, child1, child2]
    max_f = -1
    max_vec = []
    for sample in new_population:
        fitness = get_fitness(tuple(sample))
        if fitness > max_f:
            max_f = fitness
            max_vec = sample
 #   print(max_f)
    print(max_vec)


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        # Insert your code here...
        max_actions = []
        best_value = -np.inf
        for action in state.get_possible_actions(player_index=self.player_index):
            turn_next_state = MinimaxAgent.TurnBasedGameState(state, action)
            min_max_value = self.alpha_beta_value(turn_next_state, MinimaxAgent.Turn.AGENT_TURN, 2, float('-inf'),
                                                  float('inf'))
            if min_max_value > best_value:
                best_value = min_max_value
                max_actions = [action]
            elif min_max_value == best_value:
                max_actions.append(action)
        return np.random.choice(max_actions)

    def alpha_beta_value(self, state: MinimaxAgent.TurnBasedGameState, agent_to_play, depth, alpha, beta):
        if state.game_state.is_terminal_state or depth == 0:
            return heuristic(state.game_state, self.player_index)
        turn = state.turn
        if turn == agent_to_play:
            cur_max = float('-inf')
            for action in state.game_state.get_possible_actions(self.player_index):
                state.agent_action = action
                v = self.alpha_beta_value(state, agent_to_play, depth, alpha, beta)
                cur_max = max(v, cur_max)
                alpha = max(cur_max, alpha)
                if cur_max >= beta:
                    return float('inf')
            return cur_max
        else:
            cur_min = float('inf')
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                turn_next_state = MinimaxAgent.TurnBasedGameState(next_state, None)
                v = self.alpha_beta_value(turn_next_state, agent_to_play, depth - 1, alpha, beta)
                cur_min = min(v, cur_min)
                if cur_min <= alpha:
                    return float('-inf')
            return cur_min


if __name__ == '__main__':
    SAHC_sideways()
    local_search()

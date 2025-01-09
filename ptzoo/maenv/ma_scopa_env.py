from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from gymnasium import spaces
import numpy as np
import random
import itertools

NUM_ITERS = 100  # Number of iterations before truncation
PRINT_DEBUG = False

class Card:
    def __init__(self, rank: int, suit: str):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        rank_raster = self.rank

        if rank_raster == 10:
            rank_raster = "King"
        elif rank_raster == 9:
            rank_raster = "Queen"
        elif rank_raster == 8:
            rank_raster = "Jack"

        if self.suit == "bello":
            return f"{self.rank} {self.suit}"
        else:
            return f"{self.rank} di {self.suit}"

class Deck:
    suits = ['picche', 'bello', 'fiori', 'cuori']
    ranks = list(range(1, 11))  # Ranks from 1 to 7, plus 8, 9, and 10 for face cards.

    def __init__(self):
        self.cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_cards: int):
        return [self.cards.pop() for _ in range(num_cards)]

class Player:
    def __init__(self, side: int):
        self.side = side
        self.hand = []
        self.captures = []
        self.scopas = 0

    def reset(self):
        
        self.hand = []
        self.captures = []
        self.scopas = 0

    def capture(self, cards, _with):
        self.captures.extend(cards)
        if _with and _with in self.hand:
            self.hand.remove(_with)

    def play_card(self, card_index):
        return self.hand.pop(card_index)

class ScopaGame:
    def __init__(self):
        self.deck = Deck()
        self.players = [Player(1), Player(2), Player(1), Player(2)]
        self.table = []
        self.last_capture = None

    def reset(self):
        self.deck = Deck()
        self.table = []
        for player in self.players:
            player.reset()
        for player in self.players:
            player.hand = self.deck.deal(10)

    def card_in_table(self, card):
        current_table = self.table
        for comb in itertools.chain.from_iterable(itertools.combinations(current_table, r) for r in range(1, len(current_table)+1)):
            if sum(c.rank for c in comb) == card.rank:
                return True, list(comb)
        return False, []

    def play_card(self, card, player):
        isin, comb = self.card_in_table(card)
        if isin:
            for c in comb:
                self.table.remove(c)
            comb.append(card)
            player.capture(comb, _with=card)
            if not self.table:
                player.scopas += 1
        else:
            player.hand.remove(card)
            self.table.append(card)

    def evaluate_round(self):
        # Shared captures by team
        team1_captures = [card for player in self.players if player.side == 1 for card in player.captures]
        team2_captures = [card for player in self.players if player.side == 2 for card in player.captures]

        # Initialize points
        team1_points = 0
        team2_points = 0

        # Count Scopas
        team1_points += sum(player.scopas for player in self.players if player.side == 1)
        team2_points += sum(player.scopas for player in self.players if player.side == 2)

        # Most Cards
        if len(team1_captures) > len(team2_captures):
            team1_points += 1
        elif len(team2_captures) > len(team1_captures):
            team2_points += 1

        # Most Coins ("ori")
        team1_coins = [card for card in team1_captures if card.suit == 'bello']
        team2_coins = [card for card in team2_captures if card.suit == 'bello']
        if len(team1_coins) > len(team2_coins):
            team1_points += 1
        elif len(team2_coins) > len(team1_coins):
            team2_points += 1

        # Sette Bello (Seven of Coins)
        for card in team1_captures:
            if card.rank == 7 and card.suit == 'bello':
                team1_points += 1
                break
        for card in team2_captures:
            if card.rank == 7 and card.suit == 'bello':
                team2_points += 1
                break

        # Primiera
        suit_priority = {7: 4, 6: 3, 1: 2, 5: 1, 4: 0, 3: 0, 2: 0}
        team1_best_cards = [max((card for card in team1_captures if card.suit == suit), key=lambda c: suit_priority.get(c.rank, 0), default=None) for suit in Deck.suits]
        team2_best_cards = [max((card for card in team2_captures if card.suit == suit), key=lambda c: suit_priority.get(c.rank, 0), default=None) for suit in Deck.suits]

        team1_primiera = sum(suit_priority.get(card.rank, 0) for card in team1_best_cards if card)
        team2_primiera = sum(suit_priority.get(card.rank, 0) for card in team2_best_cards if card)

        if team1_primiera > team2_primiera:
            team1_points += 1
        elif team2_primiera > team1_primiera:
            team2_points += 1

        # Return final round scores
        if team1_points > team2_points:
            return 1, -1
        elif team2_points > team1_points:
            return -1, 1
        else:
            return 0, 0

class MaScopaEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "scopa_v0",
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.game = ScopaGame()
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        self._action_spaces = {
            agent: spaces.Box(0, 1, shape=(1,40)) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Box(0, 1, shape=(3, 40), dtype=np.float32) for agent in self.possible_agents
        }

        self.reset()

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def observe(self, agent):
        player_index = self.agent_name_mapping[agent]
        player = self.game.players[player_index]
        state = np.zeros((3, 40))

        for card in player.hand:
            index = (card.rank - 1) + {
                'picche': 0,
                'bello': 10,
                'fiori': 20,
                'cuori': 30
            }[card.suit]
            state[0][index] = 1

        for card in self.game.table:
            index = (card.rank - 1) + {
                'picche': 0,
                'bello': 10,
                'fiori': 20,
                'cuori': 30
            }[card.suit]
            state[1][index] = 1

        for card in player.captures:
            index = (card.rank - 1) + {
                'picche': 0,
                'bello': 10,
                'fiori': 20,
                'cuori': 30
            }[card.suit]
            state[2][index] = 1

        return state

    def reset(self):
        self.game.reset()

        self.num_moves = 0


        # Randomize the starting player SUPER IMPORTANT otherwise the not-starting side would have an advantage
        randstart = random.randint(0, 3)
        self.possible_agents = self.possible_agents[randstart:] + self.possible_agents[:randstart]
        self.agents = self.possible_agents[:]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"action_mask": self._get_action_mask(agent)} for agent in self.agents}
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def _get_action_mask(self, agent):
        player_index = self.agent_name_mapping[agent]
        player = self.game.players[player_index]
        action_mask = np.zeros(40, dtype=int)

        for card in player.hand:
            index = (card.rank - 1) + {
                'picche': 0,
                'bello': 10,
                'fiori': 20,
                'cuori': 30
            }[card.suit]
            action_mask[index] = 1

        return action_mask

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player_index = self.agent_name_mapping[agent]
        player = self.game.players[player_index]

        card = None

        for c in player.hand:
            ind = (c.rank - 1) + {
                'picche': 0,
                'bello': 10,
                'fiori': 20,
                'cuori': 30
            }[c.suit]

            if ind == action:
                card = c
                break

        self.game.play_card(card, player)

        # Check if all players have played their cards
        if all(len(player.hand) == 0 for player in self.game.players):
            # Evaluate the round and assign rewards
            round_scores = self.game.evaluate_round()
            if PRINT_DEBUG: print('round scores:', round_scores)
            for i, agent in enumerate(self.possible_agents):
                self.rewards[agent] = round_scores[self.agent_name_mapping[agent] % 2]
            if PRINT_DEBUG: print('rewards after termination?', self.rewards)
            self.terminations = {agent: True for agent in self.agents}  # End the game

        self.observations[agent] = self.observe(agent)
        self.infos[agent]["action_mask"] = self._get_action_mask(agent)
        self.num_moves += 1

        if self.num_moves >= NUM_ITERS:
            self.truncations = {a: True for a in self.agents}

        self.agent_selection = self._agent_selector.next()

    def render(self):
        if self.render_mode == "human":
            print(self.game.table)

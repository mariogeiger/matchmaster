'''
Reinforcement learning for the game MatchMaster

52 cards, 13 x 4 colors
encoded with int from 0 to 51
0 to 12 are the trumps (0 weaker 12 stronger)
13 to 25, 26 to 38 and 39 to 51 are the 3 other colors
'''
# pylint: disable=C,E1101,W0221
import torch
import torch.nn as nn
import random
import itertools
import argparse
import os
import shutil
import time_logging

def allowed_moves(table, hand):
    '''
    Given the cards in hand and the one lying on the table
    returns the allowed card to play
    '''
    if len(table) == 0:
        return hand

    same_color = [x for x in hand if x // 13 == table[0] // 13]
    if len(same_color) > 0:
        return same_color

    trumps = [x for x in hand if x // 13 == 0]

    if len(trumps) > 0:
        table_trumps = [x for x in table if x // 13 == 0]
        if len(table_trumps) > 0:
            best_table_trump = max(table_trumps)
            better_trumps = [x for x in trumps if x > best_table_trump]

            if len(better_trumps) > 0:
                return better_trumps
            return trumps
        return trumps
    return hand


def winner(table):
    '''
    Given the card on the table, returns the index of the winning card
    '''
    c = table[0]
    w = 0
    for i in range(1, len(table)):
        if (table[i] < 13 and c >= 13) or (table[i] // 13 == c // 13 and table[i] > c):
            c = table[i]
            w = i
    return w


def orthogonal_(tensor, gain=1):
    '''
    Orthogonal initialization (modified version from PyTorch)
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new_empty(rows, cols).normal_(0, 1)

    for i in range(0, rows, cols):
        # Compute the qr factorization
        q, r = torch.qr(flattened[i:i + cols].t())
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        q *= torch.diag(r, 0).sign()
        q.t_()

        with torch.no_grad():
            tensor[i:i + cols].view_as(q).copy_(q)

    with torch.no_grad():
        tensor.mul_(gain)
    return tensor


def linear(in_features, out_features, bias=True):
    '''
    Linear Module initialized properly
    '''
    m = nn.Linear(in_features, out_features, bias=bias)
    orthogonal_(m.weight)
    nn.init.zeros_(m.bias)
    return m


class Player:
    '''
    A player with its hand of cards and its memory for the LSTM
    '''
    def __init__(self, policy_bet, policy_play, hand, idx, device):
        self.policy_bet = policy_bet
        self.policy_play = policy_play
        self.hand = hand
        self.idx = idx
        self.device = device

        self.memory = None
        self.xbets = None
        self.xact = 0

    def __repr__(self):
        return "[{}: {}]".format(self.idx, self.hand)

    def bet(self, num_player, who_beggins):
        # hand + number of player + who beggins  ===>  the bet: 0, 1, ..., 7
        assert self.xbets is None
        x = torch.ones(52 + 6 + 7, device=self.device).neg()
        for i in self.hand:
            x[i] = 1
        x[52 + num_player - 2] = 1  # from 2 to 7
        x[52 + 6 + (who_beggins - self.idx) % num_player] = 1  # relative to the player
        x = self.policy_bet(x.view(1, -1))[0]
        self.xbets = x
        return x.argmax().item()

    def play(self, table, bets, num_hands):
        # hand + table + bets (yours and others) + number of won hands (yours and others)  ===>  the played card
        assert len(bets) == len(num_hands)
        x = torch.ones(52 + 52 * 6 + 8 * (1 + 6) + 8 * (1 + 6), device=self.device).neg()
        for i in self.hand:
            x[i] = 1
        r = 52
        for i in range(6):
            if i < len(table):
                x[r + table[i]] = 1
            r += 52
        for i in range(7):
            if i < len(bets):
                x[r + bets[(self.idx + i) % len(bets)]] = 1
            r += 8
        for i in range(7):
            if i < len(num_hands):
                x[r + num_hands[(self.idx + i) % len(num_hands)]] = 1
            r += 8

        x = x.view(1, 1, -1)
        x, self.memory = self.policy_play(x, self.memory)
        x = x[0, 0]

        am = x.new_zeros(52)
        am[allowed_moves(table, self.hand)] = 1 - x.min().item()
        x = x + am

        self.xact += x.max()
        c = x.argmax().item()
        if c not in self.hand:
            print("{} in {} in {}".format(c, allowed_moves(table, self.hand), self.hand))
        self.hand.remove(c)
        return c


def play_and_train(p_bet, p_play, optim, np, nc, device):
    t = time_logging.start()
    cards = list(range(52))
    random.shuffle(cards)
    players = [Player(p_bet, p_play, cards[i * nc: (i + 1) * nc], i, device) for i in range(np)]

    beg = 0
    bets = [p.bet(np, beg) for p in players]
    nhands = [0] * np
    for i in range(nc):
        table = []
        for j in range(np):
            c = players[(beg + j) % np].play(table, bets, nhands)
            table.append(c)
        w = winner(table)
        nhands[w] += 1
        beg = w

    points = [-abs(b - h) if b != h else 10 + b for b, h in zip(bets, nhands)]
    avg = sum(points) / np

    t = time_logging.end("play", t)

    loss = -sum((r - avg) * p.xact / nc + p.xbets[h] for r, p, h in zip(points, players, nhands)) / np
    optim.zero_grad()
    loss.backward()
    optim.step()

    t = time_logging.end("backward", t)
    return avg


class LSTMPolicy(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()

        self.fc_in = nn.Sequential(
            linear(in_features, hidden_features), nn.ReLU(),
            linear(hidden_features, hidden_features), nn.ReLU(),
        )

        self.lstm = nn.LSTM(hidden_features, hidden_features)

        self.fc_out = nn.Sequential(
            nn.ReLU(),
            linear(hidden_features, hidden_features), nn.ReLU(),
            linear(hidden_features, out_features),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, hc=None):
        # x (seq_len, batch, in_features)
        seq_len, batch, _ = x.size()
        x = self.fc_in(x.view(seq_len * batch, -1))
        x, hc = self.lstm(x.view(seq_len, batch, -1), hc)  # (seq_len, batch, hidden_features)
        x = self.fc_out(x.view(seq_len * batch, -1))  # (seq_len * batch, out_features)
        x = x.view(seq_len, batch, -1)  # (seq_len, batch, out_features)
        return x, hc


class GRUPolicy(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()

        self.fc_in = nn.Sequential(
            linear(in_features, hidden_features), nn.ReLU(),
        )

        self.gru = nn.GRU(hidden_features, hidden_features)

        self.fc_out = nn.Sequential(
            nn.ReLU(),
            linear(hidden_features, out_features),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, h=None):
        # x (seq_len, batch, in_features)
        seq_len, batch, _ = x.size()
        x = self.fc_in(x.view(seq_len * batch, -1))
        x, h = self.gru(x.view(seq_len, batch, -1), h)  # (seq_len, batch, hidden_features)
        x = self.fc_out(x.view(seq_len * batch, -1))  # (seq_len * batch, out_features)
        x = x.view(seq_len, batch, -1)  # (seq_len, batch, out_features)
        return x, h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--restore", type=str)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    os.mkdir(args.log_dir)
    shutil.copyfile(__file__, os.path.join(args.log_dir, "script.py"))

    policy = (
        # Bet policy
        nn.Sequential(
            linear(65, 128), nn.ReLU(),
            linear(128, 128), nn.ReLU(), 
            linear(128, 128), nn.ReLU(), 
            linear(128, 7), nn.ConstantPad1d((1, 0), 0), nn.LogSoftmax(dim=1)
        ).to(device),
        # Play policy
        LSTMPolicy(476, 256, 52).to(device)
    )

    if args.restore:
        bet, play, avgs = torch.load(args.restore, map_location=device)
        try:
            policy[0].load_state_dict(bet)
        except RuntimeError:
            pass
        policy[1].load_state_dict(play)
    else:
        avgs = []

    optim = torch.optim.Adam(list(policy[0].parameters()) + list(policy[1].parameters()))

    for i in itertools.count(1):
        np = random.randint(2, 7)  # number of players
        nc = random.randint(1, 7)  # number of cards
        t = time_logging.start()
        avg = play_and_train(*policy, optim, np, nc, device)
        time_logging.end("play & train", t)
        avgs.append(avg)
        print("{} {:.1f}   ".format(i, sum(avgs[-200:]) / len(avgs[-200:])), end="\r")

        if i % 1000 == 0:
            torch.save((policy[0].state_dict(), policy[1].state_dict(), avgs), os.path.join(args.log_dir, "save.pkl"))

            print(time_logging.text_statistics())


if __name__ == "__main__":
    main()

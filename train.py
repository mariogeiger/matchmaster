# pylint: disable=C,E1101,W0221
import torch
import torch.nn as nn
import random
import itertools
import argparse
import os
import shutil


def allowed_moves(table, hand):
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
    c = table[0]
    w = 0
    for i in range(1, len(table)):
        if (table[i] < 13 and c >= 13) or (table[i] // 13 == c // 13 and table[i] > c):
            c = table[i]
            w = i
    return w


def orthogonal_(tensor, gain=1):
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
    m = nn.Linear(in_features, out_features, bias=bias)
    orthogonal_(m.weight)
    nn.init.zeros_(m.bias)
    return m


class Player:
    def __init__(self, policy_bet, policy_play, hand, idx):
        self.policy_bet = policy_bet
        self.policy_play = policy_play
        self.idx = idx
        self.hand = hand

        self.memory = None
        self.xbet = None
        self.xact = 0

    def __repr__(self):
        return "[{}: {}]".format(self.idx, self.hand)

    def bet(self, num_player, who_beggins):
        # hand + number of player + who beggins  ===>  the bet: 0, 1, ..., 7
        assert self.xbet is None
        x = torch.ones(52 + 6 + 7).neg()
        for i in self.hand:
            x[i] = 1
        x[52 + num_player - 2] = 1  # from 2 to 7
        x[52 + 6 + (who_beggins - self.idx) % num_player] = 1  # relative to the player
        x = self.policy_bet(x.view(1, -1))[0]
        self.xbet = x.max()
        return x.argmax().item()

    def play(self, table, bets, num_hands):
        # hand + table + bets (yours and others) + number of won hands (yours and others)  ===>  the played card
        assert len(bets) == len(num_hands)
        x = torch.ones(52 + 52 * 6 + 8 * (1 + 6) + 8 * (1 + 6)).neg()
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

        # x = x - torch.rand(52)  # add randomness to explore more

        am = torch.zeros(52)
        am[allowed_moves(table, self.hand)] = 1 - x.min().item()
        x = x + am

        self.xact += x.max()
        c = x.argmax().item()
        self.hand.remove(c)
        return c


def play_and_train(p_bet, p_play, optim, np, nc):
    cards = list(range(52))
    random.shuffle(cards)
    players = [Player(p_bet, p_play, cards[i * nc: (i + 1) * nc], i) for i in range(np)]

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

    loss = -sum((r - avg) * (p.xbet + p.xact / nc) for r, p in zip(points, players)) / np
    optim.zero_grad()
    loss.backward()
    optim.step()

    return avg


class PlayPolicy(nn.Module):
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


class GRUDeepPolicy(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()

        self.fc_in = nn.Sequential(
            linear(in_features, (in_features + hidden_features) // 2), nn.ReLU(),
            linear((in_features + hidden_features) // 2, hidden_features), nn.ReLU(),
            linear(hidden_features, hidden_features), nn.ReLU(),
        )

        self.gru = nn.GRU(hidden_features, hidden_features)

        self.fc_out = nn.Sequential(
            nn.ReLU(),
            linear(hidden_features, hidden_features), nn.ReLU(),
            linear(hidden_features, hidden_features), nn.ReLU(),
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
    args = parser.parse_args()
    os.mkdir(args.log_dir)
    shutil.copyfile(__file__, os.path.join(args.log_dir, "script.py"))

    # hand + number of player + who beggins  ===>  the bet: 0, 1, ..., 7
    p_bet = nn.Sequential(linear(52 + 6 + 7, 8), nn.LogSoftmax(dim=1))

    # hand + table + bets (yours and others) + number of won hands (yours and others)  ===>  the played card
    p_play = GRUDeepPolicy(52 + 52 * 6 + 8 * (1 + 6) + 8 * (1 + 6), 64, 52)

    optim = torch.optim.Adam(list(p_bet.parameters()) + list(p_play.parameters()))

    avgs = []
    for i in itertools.count():
        np = random.randint(2, 7)  # number of players
        nc = random.randint(1, 7)  # number of cards
        avg = play_and_train(p_bet, p_play, optim, np, nc)
        avgs.append(avg)
        print("{} {:.1f}   ".format(i, avg), end="\r")

        if i % 1000 == 0:
            torch.save((p_bet.state_dict(), p_play.state_dict(), avgs), os.path.join(args.log_dir, "save.pkl"))


if __name__ == "__main__":
    main()

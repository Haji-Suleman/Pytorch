import pygame
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------- GAME ENVIRONMENT -------------------
class SnakeGame:
    def __init__(self, w=200, h=200, cell=20, render=False):
        self.w, self.h = w, h
        self.cell = cell
        self.render_game = render
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((w, h))
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(self.w//2, self.h//2)]
        self.dx, self.dy = self.cell, 0
        self.place_food()
        self.done = False
        self.score = 0
        return self.get_state()

    def place_food(self):
        self.food = (random.randrange(0, self.w, self.cell),
                     random.randrange(0, self.h, self.cell))
        while self.food in self.snake:
            self.food = (random.randrange(0, self.w, self.cell),
                         random.randrange(0, self.h, self.cell))

    def step(self, action):

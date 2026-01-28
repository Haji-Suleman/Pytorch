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
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(self.w // 2, self.h // 2)]
        self.dx, self.dy = self.cell, 0
        self.place_food()
        self.done = False
        self.score = 0
        return self.get_state()

    def place_food(self):
        self.food = (
            random.randrange(0, self.w, self.cell),
            random.randrange(0, self.h, self.cell),
        )
        while self.food in self.snake:
            self.food = (
                random.randrange(0, self.w, self.cell),
                random.randrange(0, self.h, self.cell),
            )

    def step(self, action):
        # Action: 0=UP,1=DOWN,2=LEFT,3=RIGHT
        if action == 0 and self.dy == 0:
            self.dx, self.dy = 0, -self.cell
        if action == 1 and self.dy == 0:
            self.dx, self.dy = 0, self.cell
        if action == 2 and self.dx == 0:
            self.dx, self.dy = -self.cell, 0
        if action == 3 and self.dx == 0:
            self.dx, self.dy = self.cell, 0

        head = (self.snake[0][0] + self.dx, self.snake[0][1] + self.dy)

        # Collision check
        if (
            head[0] < 0
            or head[0] >= self.w
            or head[1] < 0
            or head[1] >= self.h
            or head in self.snake
        ):
            self.done = True
            reward = -1
            return self.get_state(), reward, self.done

        self.snake.insert(0, head)
        reward = 0

        if head == self.food:
            reward = 1
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

        return self.get_state(), reward, self.done

    def get_state(self):
        head = self.snake[0]
        food_x, food_y = self.food
        dx, dy = self.dx, self.dy
        return np.array([head[0], head[1], food_x, food_y, dx, dy], dtype=np.float32)

    def render(self):
        if not self.render_game:
            return
        self.screen.fill((0, 0, 0))
        for s in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (*s, self.cell, self.cell))
        pygame.draw.rect(self.screen, (255, 0, 0), (*self.food, self.cell, self.cell))
        pygame.display.flip()
        self.clock.tick(10)


# ------------------- DQN AGENT -------------------
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.fc(x)


# ------------------- TRAINING -------------------
def train_snake(episodes=500, render_every=100):
    env = SnakeGame(render=False)
    state_size = 6
    action_size = 4
    hidden_size = 128
    lr = 0.001
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 64
    memory = deque(maxlen=2000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DQN(state_size, hidden_size, action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for ep in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).to(device)
        total_reward = 0
        steps = 0

        while True:
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state)).item()

            next_state, reward, done = env.step(action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
            memory.append((state, action, reward, next_state_tensor, done))
            state = next_state_tensor
            total_reward += reward
            steps += 1

            if done:
                break

            # Train from memory
            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                states_mb = torch.stack([s[0] for s in minibatch])
                actions_mb = torch.tensor([s[1] for s in minibatch]).to(device)
                rewards_mb = torch.tensor(
                    [s[2] for s in minibatch], dtype=torch.float32
                ).to(device)
                next_states_mb = torch.stack([s[3] for s in minibatch])
                dones_mb = torch.tensor(
                    [s[4] for s in minibatch], dtype=torch.float32
                ).to(device)

                q_values = (
                    model(states_mb).gather(1, actions_mb.unsqueeze(1)).squeeze(1)
                )
                with torch.no_grad():
                    q_next = model(next_states_mb).max(1)[0]
                    q_target = rewards_mb + gamma * q_next * (1 - dones_mb)
                loss = criterion(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % 10 == 0:
            print(
                f"Episode {ep} - Reward: {total_reward} - Steps: {steps} - Epsilon: {epsilon:.2f}"
            )

        if ep % render_every == 0:
            env.render()  # optional small render to watch progress

    print("Training finished.")
    return model


# ------------------- PLAY FUNCTION -------------------
def play_snake(model, episodes=5):
    env = SnakeGame(render=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ep in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).to(device)
        done = False
        while not done:
            pygame.event.pump()  # prevent pygame from freezing
            with torch.no_grad():
                action = torch.argmax(model(state)).item()
            next_state, reward, done = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32).to(device)
            env.render()
        print(f"Episode {ep+1} finished with score: {env.score}")
    pygame.quit()


# ------------------- MAIN -------------------
if __name__ == "__main__":
    trained_model = train_snake(episodes=300)  # train the AI
    play_snake(trained_model, episodes=3)  # watch AI play

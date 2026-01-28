import pygame
import random

pygame.init()

w, h = 600, 400
screen = pygame.display.set_mode((w, h))
clock = pygame.time.Clock()


cell = 20
snake = [(100, 100), (80, 100), (60, 100)]
dx, dy = cell, 0
food = (random.randrange(0, w, cell), random.randrange(0, h, cell))

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_UP and dy == 0:
                dx, dy = 0, -cell
            if e.key == pygame.K_DOWN and dy == 0:
                dx, dy = 0, cell
            if e.key == pygame.K_LEFT and dx == 0:
                dx, dy = -cell, 0
            if e.key == pygame.K_RIGHT and dx == 0:
                dx, dy = cell, 0

    head = (snake[0][0] + dx, snake[0][1] + dy)

    if head[0] < 0 or head[0] >= w or head[1] < 0 or head[1] >= h or head in snake:
        break

    snake.insert(0, head)

    if head == food:
        food = (random.randrange(0, w, cell), random.randrange(0, h, cell))
    else:
        snake.pop()

    screen.fill((0, 0, 0))
    for s in snake:
        pygame.draw.rect(screen, (0, 255, 0), (*s, cell, cell))
    pygame.draw.rect(screen, (255, 0, 0), (*food, cell, cell))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()

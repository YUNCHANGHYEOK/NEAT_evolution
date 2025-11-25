import pygame
import neat
import os
import random
import math

# --- 1. 전역 설정 ---
WIN_WIDTH = 800
WIN_HEIGHT = 600
GEN = 0
FPS = 60
MAX_GEN_TIME = 1800
FOOD_COUNT = 5

# --------------------------------
# 종 ID에 기반한 색상을 생성
# --------------------------------
def get_color_from_id(id):
    """Species ID를 기반으로 고유한 기본 RGB 색상 생성"""
    if id <= 0:
        return (0, 150, 0)

    random.seed(id)
    r = random.randint(50, 200)
    g = random.randint(50, 200)
    b = random.randint(50, 200)
    random.seed(None)

    return (r, g, b)

# --------------------------------
# 먹이 클래스
# --------------------------------
class Food:
    def __init__(self):
        self.x = random.randint(50, WIN_WIDTH - 50)
        self.y = random.randint(50, WIN_HEIGHT - 50)
        self.rad = 10

    def draw(self, win):
        pygame.draw.circle(win, (255, 0, 0), (self.x, self.y), self.rad)

# --------------------------------
# 생명체 클래스
# --------------------------------
class Creature:
    def __init__(self, species_id=0):
        self.x = WIN_WIDTH / 2
        self.y = WIN_HEIGHT / 2

        self.vel = 5
        self.rect = pygame.Rect(self.x, self.y, 20, 20)
        self.life = 600

        self.species_id = species_id
        self.base_color = get_color_from_id(species_id)

    def move(self, output):
        if output[0] > 0.5:
            self.y -= self.vel
        if output[1] > 0.5:
            self.y += self.vel
        if output[2] > 0.5:
            self.x -= self.vel
        if output[3] > 0.5:
            self.x += self.vel

        self.x = max(0, min(self.x, WIN_WIDTH - 20))
        self.y = max(0, min(self.y, WIN_HEIGHT - 20))

        self.rect.topleft = (self.x, self.y)

    def draw(self, win, font):
        life_ratio = min(1.0, self.life / 600.0)

        r = int(self.base_color[0] * life_ratio)
        g = int(self.base_color[1] * life_ratio)
        b = int(self.base_color[2] * life_ratio)

        r = max(10, r)
        g = max(10, g)
        b = max(10, b)

        current_color = (r, g, b)

        pygame.draw.rect(win, current_color, self.rect)

        life_text = font.render(str(int(self.life)), True, (255, 255, 255))
        win.blit(life_text, life_text.get_rect(center=self.rect.center))

# --------------------------------
# genome 평가 함수
# --------------------------------
def eval_genomes(genomes, config):
    global GEN
    GEN += 1

    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    nets = []
    creatures = []
    ge = []

    foods = [Food() for _ in range(FOOD_COUNT)]

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

        species_id = getattr(genome, "species_id", 1)
        creatures.append(Creature(species_id))

        ge.append(genome)

    run = True
    total_time = 0

    info_font = pygame.font.SysFont("comicsans", 30)
    creature_font = pygame.font.SysFont("comicsans", 16)

    while run and len(creatures) > 0:
        total_time += 1
        if total_time > MAX_GEN_TIME:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        for i in reversed(range(len(creatures))):
            creature = creatures[i]

            closest_food = None
            min_distance = float("inf")

            for f in foods:
                distance = math.sqrt((creature.x - f.x)**2 + (creature.y - f.y)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_food = f

            if closest_food:
                dx = closest_food.x - creature.x
                dy = closest_food.y - creature.y
                output = nets[i].activate((dx, dy))
            else:
                output = nets[i].activate((0, 0))

            creature.move(output)

            creature.life -= 1
            if creature.life <= 0:
                creatures.pop(i)
                nets.pop(i)
                ge.pop(i)
                continue

            ge[i].fitness += 0.1

            if closest_food and min_distance < 20:
                ge[i].fitness += 20
                creature.life += 300

                foods.remove(closest_food)
                foods.append(Food())

        screen.fill((0, 0, 0))

        for food_item in foods:
            food_item.draw(screen)

        for creature in creatures:
            creature.draw(screen, creature_font)

        remain_time = max(0, (MAX_GEN_TIME - total_time) // FPS)
        unique_species = len(set(c.species_id for c in creatures))

        info_text = info_font.render(
            f"Gen: {GEN} | Alive: {len(creatures)} | Species: {unique_species} | Time Left: {remain_time}s",
            True,
            (255, 255, 255)
        )
        screen.blit(info_text, (10, 10))

        pygame.display.update()
        clock.tick(FPS)

# --------------------------------
# NEAT 실행
# --------------------------------
def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)

# --------------------------------
# 메인
# --------------------------------
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)

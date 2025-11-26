import pygame
import neat
import os
import random
import math

# --- 1. 전역 설정 ---
WIN_WIDTH = 800
WIN_HEIGHT = 600
INFO_PANEL_WIDTH = 300
GEN = 0
FPS = 60
MAX_GEN_TIME = 600
FOOD_COUNT = 3

# --- Fitness 로그 저장 ---
fitness_log_best = []
fitness_log_avg = []

# --- 선택된 Generation ---
selected_gen = None  # None이면 선택 안함
selected_gen_box_height = 80


# --------------------------------
# 종 ID에 기반한 색상 생성
# --------------------------------
def get_color_from_id(id):
    if id <= 0:
        return (0, 150, 0)

    random.seed(id)
    r = random.randint(50, 200)
    g = random.randint(50, 200)
    b = random.randint(50, 200)
    random.seed(None)
    return (r, g, b)


# --------------------------------
# 먹이
# --------------------------------
class Food:
    def __init__(self):
        self.x = random.randint(50, WIN_WIDTH - 50)
        self.y = random.randint(50, WIN_HEIGHT - 50)
        self.rad = 10

    def draw(self, win):
        pygame.draw.circle(win, (255, 0, 0), (self.x, self.y), self.rad)


# --------------------------------
# 생명체
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
        if output[0] > 0.5: self.y -= self.vel
        if output[1] > 0.5: self.y += self.vel
        if output[2] > 0.5: self.x -= self.vel
        if output[3] > 0.5: self.x += self.vel

        self.x = max(0, min(self.x, WIN_WIDTH - 20))
        self.y = max(0, min(self.y, WIN_HEIGHT - 20))
        self.rect.topleft = (self.x, self.y)

    def draw(self, win, font):
        life_ratio = min(1.0, self.life / 600.0)
        r = max(10, int(self.base_color[0] * life_ratio))
        g = max(10, int(self.base_color[1] * life_ratio))
        b = max(10, int(self.base_color[2] * life_ratio))

        pygame.draw.rect(win, (r, g, b), self.rect)
        life_text = font.render(str(int(self.life)), True, (255, 255, 255))
        win.blit(life_text, life_text.get_rect(center=self.rect.center))


# --------------------------------
# 패널 정보
# --------------------------------
def draw_info_panel(screen, font, gen, alive, best_fit, avg_fit, species_count, remain_time):
    panel_x = WIN_WIDTH
    pygame.draw.rect(screen, (40, 40, 40), (panel_x, 0, INFO_PANEL_WIDTH, WIN_HEIGHT))

    texts = [
        f"Generation: {gen}",
        f"Alive: {alive}",
        f"Best Fitness: {best_fit:.2f}",
        f"Avg Fitness: {avg_fit:.2f}",
        f"Species: {species_count}",
        f"Time Left: {remain_time}s",
    ]

    y = 20
    for t in texts:
        txt = font.render(t, True, (255, 255, 255))
        screen.blit(txt, (panel_x + 20, y))
        y += 35


# --------------------------------
# 선택된 generation 정보 박스
# --------------------------------
def draw_selected_gen_box(screen, font):
    global selected_gen
    if selected_gen is None:
        return

    panel_x = WIN_WIDTH
    box_y = WIN_HEIGHT - selected_gen_box_height

    pygame.draw.rect(screen, (50, 50, 50), (panel_x, box_y, INFO_PANEL_WIDTH, selected_gen_box_height))

    best = fitness_log_best[selected_gen]
    avg = fitness_log_avg[selected_gen]

    text1 = font.render(f"Selected Gen: {selected_gen}", True, (255, 255, 255))
    text2 = font.render(f"Best: {best:.2f}", True, (255, 255, 0))
    text3 = font.render(f"Avg: {avg:.2f}", True, (0, 180, 255))

    screen.blit(text1, (panel_x + 20, box_y + 10))
    screen.blit(text2, (panel_x + 20, box_y + 35))
    screen.blit(text3, (panel_x + 140, box_y + 35))


# --------------------------------
# 그래프
# --------------------------------
def draw_fitness_graph(screen):
    global selected_gen

    if len(fitness_log_best) < 2:
        return

    panel_x = WIN_WIDTH
    graph_x = panel_x + 20
    graph_y = 260
    graph_w = INFO_PANEL_WIDTH - 40
    graph_h = 210

    pygame.draw.rect(screen, (60, 60, 60), (graph_x, graph_y, graph_w, graph_h), 2)

    max_val = max(max(fitness_log_best), max(fitness_log_avg))
    if max_val == 0:
        max_val = 1

    step_x = graph_w / (len(fitness_log_best) - 1)

    # best
    points_best = []
    for i, v in enumerate(fitness_log_best):
        x = graph_x + i * step_x
        y = graph_y + graph_h - (v / max_val) * graph_h
        points_best.append((x, y))
    pygame.draw.lines(screen, (255, 255, 0), False, points_best, 2)

    # avg
    points_avg = []
    for i, v in enumerate(fitness_log_avg):
        x = graph_x + i * step_x
        y = graph_y + graph_h - (v / max_val) * graph_h
        points_avg.append((x, y))
    pygame.draw.lines(screen, (0, 180, 255), False, points_avg, 2)

    # 범례
    font = pygame.font.SysFont("comicsans", 18)
    screen.blit(font.render("Best Fitness", True, (255, 255, 0)), (graph_x, graph_y - 35))
    screen.blit(font.render("Avg Fitness", True, (0, 180, 255)), (graph_x + 140, graph_y - 35))


    # 🔵 선택된 지점 마커 찍기
    if selected_gen is not None and selected_gen < len(fitness_log_best):
        marker_x = graph_x + selected_gen * step_x
        marker_y = graph_y + graph_h - (fitness_log_best[selected_gen] / max_val) * graph_h
        pygame.draw.circle(screen, (200, 200, 200), (int(marker_x), int(marker_y)), 4)


# --------------------------------
# 그래프 클릭 처리
# --------------------------------
def handle_graph_click(mouse_pos):
    global selected_gen

    panel_x = WIN_WIDTH
    graph_x = panel_x + 20
    graph_y = 230
    graph_w = INFO_PANEL_WIDTH - 40
    graph_h = 230

    mx, my = mouse_pos

    if not (graph_x <= mx <= graph_x + graph_w and graph_y <= my <= graph_y + graph_h):
        return

    step_x = graph_w / (len(fitness_log_best) - 1)
    index = int((mx - graph_x) / step_x)

    if 0 <= index < len(fitness_log_best):
        selected_gen = index


# --------------------------------
# genome 평가 함수
# --------------------------------
def eval_genomes(genomes, config):
    global GEN
    GEN += 1

    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH + INFO_PANEL_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    nets, creatures, ge = [], [], []
    foods = [Food() for _ in range(FOOD_COUNT)]

    for genome_id, genome in genomes:
        genome.fitness = 0
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        creatures.append(Creature(getattr(genome, "species_id", 1)))
        ge.append(genome)

    info_font = pygame.font.SysFont("comicsans", 24)
    creature_font = pygame.font.SysFont("comicsans", 16)

    total_time = 0
    run = True

    while run and len(creatures) > 0:
        total_time += 1
        if total_time > MAX_GEN_TIME:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                handle_graph_click(event.pos)

        for i in reversed(range(len(creatures))):
            creature = creatures[i]

            closest_food = None
            min_distance = float("inf")
            for f in foods:
                d = math.dist((creature.x, creature.y), (f.x, f.y))
                if d < min_distance:
                    min_distance = d
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

        # ---------------- 화면 업데이트 ----------------
        screen.fill((0, 0, 0))

        for food in foods: food.draw(screen)
        for c in creatures: c.draw(screen, creature_font)

        fitness_values = [g.fitness for g in ge]
        best = max(fitness_values)
        avg = sum(fitness_values) / len(fitness_values)
        species_count = len(set(c.species_id for c in creatures))
        remain = max(0, (MAX_GEN_TIME - total_time) // FPS)

        draw_info_panel(screen, info_font, GEN, len(creatures), best, avg, species_count, remain)
        draw_fitness_graph(screen)
        draw_selected_gen_box(screen, info_font)

        pygame.display.update()
        clock.tick(FPS)

    # 세대 종료 데이터 저장
    fitness_values = [g.fitness for g in ge]
    fitness_log_best.append(max(fitness_values))
    fitness_log_avg.append(sum(fitness_values) / len(fitness_values))


# --------------------------------
# 실행
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
    p.add_reporter(neat.StatisticsReporter())

    p.run(eval_genomes, 50)


if __name__ == "__main__":
    dir_path = os.path.dirname(__file__)
    config_path = os.path.join(dir_path, "config-feedforward.txt")
    run(config_path)

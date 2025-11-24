import pygame
import neat
import os
import random
import math

# --- 1. 전역 설정 ---
WIN_WIDTH = 800
WIN_HEIGHT = 600
GEN = 0 # 현재 세대 수
FPS = 60 # 프레임 (초당 60회 갱신)
MAX_GEN_TIME = 1800 # 최대 세대 유지 시간 (30초 * 60FPS)

class Food:
    def __init__(self):
        self.x = random.randint(50, WIN_WIDTH - 50)
        self.y = random.randint(50, WIN_HEIGHT - 50)
        self.rad = 10

    def draw(self, win):
        pygame.draw.circle(win, (255, 0, 0), (self.x, self.y), self.rad)

class Creature:
    def __init__(self):
        self.x = WIN_WIDTH / 2 # 시작 위치
        self.y = WIN_HEIGHT / 2
        self.vel = 5
        self.rect = pygame.Rect(self.x, self.y, 20, 20)
        self.life = 600  # 초기 수명 (10초)
        
    def move(self, output):
        # 신경망 출력값(0~1)에 따라 이동
        if output[0] > 0.5: self.y -= self.vel
        if output[1] > 0.5: self.y += self.vel
        if output[2] > 0.5: self.x -= self.vel
        if output[3] > 0.5: self.x += self.vel

        # 화면 밖으로 못 나가게 막기
        self.x = max(0, min(self.x, WIN_WIDTH - 20))
        self.y = max(0, min(self.y, WIN_HEIGHT - 20))
        self.rect.topleft = (self.x, self.y)

    def draw(self, win):
        # [수정 반영] life_ratio를 1.0으로 제한하여 255를 초과하는 색상 값이 나오지 않도록 함
        life_ratio = min(1.0, self.life / 600) 
        green_intensity = int(255 * life_ratio)
        current_color = (0, green_intensity, 0)
        
        pygame.draw.rect(win, current_color, self.rect)

def eval_genomes(genomes, config):
    global GEN
    GEN += 1
    
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    
    nets = []
    creatures = []
    ge = []
    
    food = Food()

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        creatures.append(Creature())
        ge.append(genome)

    run = True
    total_time = 0 

    while run and len(creatures) > 0:
        total_time += 1

        # 하드 타임 리밋: 30초가 지나면 무조건 세대 종료
        if total_time > MAX_GEN_TIME: 
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 리스트에서 제거 시 오류 방지를 위해 리버스 루프 사용
        for i in reversed(range(len(creatures))):
            creature = creatures[i]
            
            # 1. 입력: 먹이와 나의 거리 차이 (dx, dy)
            dx = food.x - creature.x
            dy = food.y - creature.y
            
            # 2. 신경망 판단 및 움직임
            output = nets[i].activate((dx, dy))
            creature.move(output)
            
            # 3. 수명 감소 및 사망 판정
            creature.life -= 1 # 매 프레임마다 수명 1 감소
            
            if creature.life <= 0:
                # [사망] 수명이 0 이하가 되면 리스트에서 제거
                creatures.pop(i)
                nets.pop(i)
                ge.pop(i)
                continue # 다음 생명체로 넘어감

            # 4. 생존 보너스 및 적합도 증가
            ge[i].fitness += 0.1 # 살아있는 동안 계속 점수 부여

            # 5. 먹이를 먹었는지 판별
            distance = math.sqrt((creature.x - food.x)**2 + (creature.y - food.y)**2)
            if distance < 20:
                ge[i].fitness += 20 # 먹으면 큰 점수
                creature.life += 300 # 먹이를 먹으면 수명 5초 추가
                food = Food() # 먹이 위치 이동 (새로운 목표 생성)
        
        # --- 그리기 및 정보 표시 ---
        screen.fill((0, 0, 0))
        food.draw(screen)
        
        for creature in creatures:
            creature.draw(screen)
            
        font = pygame.font.SysFont("comicsans", 30)
        remain_time = max(0, (MAX_GEN_TIME - total_time) // FPS)
        text = font.render(f"Gen: {GEN} | Alive: {len(creatures)} | Time Left: {remain_time}s", 1, (255, 255, 255))
        screen.blit(text, (10, 10))

        pygame.display.update()
        clock.tick(FPS)

def run(config_path):
    
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50) 

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
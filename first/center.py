import pygame
import neat
import os
import random
import math

# --- 설정 ---
WIN_WIDTH = 800
WIN_HEIGHT = 600
GEN = 0

class Food:
    def __init__(self):
        self.x = random.randint(50, WIN_WIDTH - 50)
        self.y = random.randint(50, WIN_HEIGHT - 50)
        self.rad = 10

    def draw(self, win):
        pygame.draw.circle(win, (255, 0, 0), (self.x, self.y), self.rad)

class Creature:
    def __init__(self):
        self.x = WIN_WIDTH / 2
        self.y = WIN_HEIGHT / 2
        self.vel = 5
        self.rect = pygame.Rect(self.x, self.y, 20, 20)
        self.color = (0, 255, 0)
        
    def move(self, output):
        # 신경망 출력값(0~1)에 따라 이동
        # output 리스트 순서: [상, 하, 좌, 우]
        if output[0] > 0.5: self.y -= self.vel
        if output[1] > 0.5: self.y += self.vel
        if output[2] > 0.5: self.x -= self.vel
        if output[3] > 0.5: self.x += self.vel

        # 화면 밖으로 못 나가게 막기
        self.x = max(0, min(self.x, WIN_WIDTH - 20))
        self.y = max(0, min(self.y, WIN_HEIGHT - 20))
        self.rect.topleft = (self.x, self.y)

    def draw(self, win):
        pygame.draw.rect(win, self.color, self.rect)

def eval_genomes(genomes, config):
    global GEN
    GEN += 1
    
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    
    # 이번 세대의 생명체와 신경망 리스트
    nets = []
    creatures = []
    ge = []
    
    # 먹이는 하나만 생성 (모든 생명체가 이걸 노림)
    food = Food()

    for genome_id, genome in genomes:
        genome.fitness = 0  # 점수 초기화
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        creatures.append(Creature())
        ge.append(genome)

    run = True
    # 최대 10초(600프레임) 동안 못 먹으면 다음 세대로 강제 종료 (무한 루프 방지)
    timer = 0 
    
    while run and len(creatures) > 0:
        timer += 1
        if timer > 600: # 시간이 너무 오래 걸리면 강제 종료
            run = False
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # --- 생명체 로직 ---
        for i, creature in enumerate(creatures):
            # 1. 입력(Input): 먹이와 나의 거리 차이 (dx, dy)
            # 설정 파일에서 num_inputs=2로 했으므로 딱 2개만 넣어야 함
            dx = food.x - creature.x
            dy = food.y - creature.y
            
            # 2. 신경망 판단
            output = nets[i].activate((dx, dy))
            creature.move(output)
            
            # 3. 생존 보너스 (조금씩이라도 점수를 줘서 살아있는걸 장려)
            ge[i].fitness += 0.1

            # 4. 먹이를 먹었는지 판별
            distance = math.sqrt((creature.x - food.x)**2 + (creature.y - food.y)**2)
            if distance < 20:
                ge[i].fitness += 10  # 먹으면 큰 점수!
                
                # 먹은 놈만 남기고 나머지는 죽일 수도 있고,
                # 먹이를 이동시킬 수도 있음. 여기선 먹이를 이동시킴.
                food = Food() 
                timer = 0 # 먹었으면 시간 연장

        # --- 그리기 ---
        screen.fill((0, 0, 0))
        food.draw(screen) # 빨간 원(먹이)
        
        for creature in creatures:
            creature.draw(screen) # 초록 네모(생명체)
            
        # 정보 표시
        font = pygame.font.SysFont("comicsans", 30)
        text = font.render(f"Gen: {GEN} | Alive: {len(creatures)}", 1, (255, 255, 255))
        screen.blit(text, (10, 10))

        pygame.display.update()
        clock.tick(60) # 속도가 너무 빠르면 30으로 낮추세요

def run(config_path):
    # 설정 파일 읽기 (UTF-8 처리 포함)
    import configparser
    p = configparser.ConfigParser()
    p.read(config_path) # 한글 주석 없으므로 기본 read로 충분
    
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50) # 50세대까지 실행

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
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
MAX_GEN_TIME = 1200 # 20초 (사용자 설정 유지)
NUM_PREDATORS = 1 # [수정 완료] 포식자 1마리
NUM_FOODS = 2 

class Food:
    def __init__(self):
        self.x = random.randint(50, WIN_WIDTH - 50)
        self.y = random.randint(50, WIN_HEIGHT - 50)
        self.rad = 10

    def draw(self, win):
        pygame.draw.circle(win, (255, 0, 0), (self.x, self.y), self.rad)

class Predator:
    def __init__(self):
        self.x = random.randint(50, WIN_WIDTH - 50)
        self.y = random.randint(50, WIN_HEIGHT - 50)
        self.vel = 5 
        self.color = (0, 0, 255)
        self.rad = 15
        
        self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.change_timer = random.randint(30, 90)

    def move(self):
        self.change_timer -= 1
        
        hit_boundary = False
        if self.x <= self.rad or self.x >= WIN_WIDTH - self.rad: hit_boundary = True
        if self.y <= self.rad or self.y >= WIN_HEIGHT - self.rad: hit_boundary = True

        if hit_boundary or self.change_timer <= 0:
            self.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            self.change_timer = random.randint(30, 90)
        
        self.x += self.direction[0] * self.vel
        self.y += self.direction[1] * self.vel

        self.x = max(self.rad, min(self.x, WIN_WIDTH - self.rad))
        self.y = max(self.rad, min(self.y, WIN_HEIGHT - self.rad))

    def draw(self, win):
        pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), self.rad)

class Creature:
    def __init__(self):
        self.x = WIN_WIDTH / 2
        self.y = WIN_HEIGHT / 2
        self.vel = 5
        self.rect = pygame.Rect(self.x, self.y, 20, 20)
        self.life = 600
        self.angle = random.uniform(0, 2 * math.pi) # [추가] 초기 방향 (라디안)
        
    def move(self, output):
        # output: [회전(좌), 회전(우), 전진, 후진]으로 해석
        rotation_rate = 0.15 
        thrust = 1.0 

        # 회전 적용
        if output[0] > 0.5: self.angle = (self.angle - rotation_rate) % (2 * math.pi) 
        if output[1] > 0.5: self.angle = (self.angle + rotation_rate) % (2 * math.pi) 

        # 전진/후진 적용
        move_x = 0
        move_y = 0
        if output[2] > 0.5: # 전진
            move_x = math.cos(self.angle) * self.vel * thrust
            move_y = math.sin(self.angle) * self.vel * thrust
        if output[3] > 0.5: # 후진
            move_x = -math.cos(self.angle) * self.vel * thrust
            move_y = -math.sin(self.angle) * self.vel * thrust

        self.x += move_x
        self.y += move_y

        # 화면 경계 처리
        self.x = max(0, min(self.x, WIN_WIDTH - 20))
        self.y = max(0, min(self.y, WIN_HEIGHT - 20))
        self.rect.topleft = (self.x, self.y)

    def draw(self, win):
        life_ratio = min(1.0, self.life / 600) 
        green_intensity = int(255 * life_ratio)
        current_color = (0, green_intensity, 0)
        pygame.draw.rect(win, current_color, self.rect)
        
        # [추가] 생명체의 방향을 나타내는 선 그리기
        line_length = 15
        end_x = self.x + 10 + math.cos(self.angle) * line_length
        end_y = self.y + 10 + math.sin(self.angle) * line_length
        pygame.draw.line(win, (255, 255, 255), (self.x + 10, self.y + 10), (end_x, end_y), 2)


def eval_genomes(genomes, config):
    global GEN
    GEN += 1
    
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    
    nets = []
    creatures = []
    ge = []
    
    foods = [Food() for _ in range(NUM_FOODS)]
    predators = [Predator() for _ in range(NUM_PREDATORS)] # NUM_PREDATORS = 1

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

        if total_time > MAX_GEN_TIME: 
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        for predator in predators:
            predator.move()

        # [핵심] 입력 처리 및 생명체 로직
        for i in reversed(range(len(creatures))):
            creature = creatures[i]
            
            # --- 1. 신경망 입력 계산 (총 4개 입력: F1_dist, F1_angle, P1_dist, P1_angle) ---
            
            # A. 가장 가까운 먹이 (F1) 정보
            food_distances = []
            for food_item in foods:
                dist = math.sqrt((food_item.x - creature.x)**2 + (food_item.y - creature.y)**2)
                food_distances.append((dist, food_item))
            food_distances.sort(key=lambda x: x[0])
            
            food1 = food_distances[0][1]
            dist_f = food_distances[0][0] # 거리
            angle_f_abs = math.atan2(food1.y - creature.y, food1.x - creature.x)
            angle_f_rel = angle_f_abs - creature.angle 
            angle_f_rel = (angle_f_rel + math.pi) % (2 * math.pi) - math.pi # 상대 각도 정규화 (-PI ~ PI)

            # B. 가장 가까운 포식자 (P1) 정보
            predator_distances = []
            for predator in predators:
                dist = math.sqrt((predator.x - creature.x)**2 + (predator.y - creature.y)**2)
                predator_distances.append((dist, predator))
            predator_distances.sort(key=lambda x: x[0])
            
            predator1 = predator_distances[0][1]
            dist_p = predator_distances[0][0] # 거리
            angle_p_abs = math.atan2(predator1.y - creature.y, predator1.x - creature.x)
            angle_p_rel = angle_p_abs - creature.angle 
            angle_p_rel = (angle_p_rel + math.pi) % (2 * math.pi) - math.pi # 상대 각도 정규화 (-PI ~ PI)
            
            # 최종 입력: (F_dist, F_angle_rel, P_dist, P_angle_rel)
            inputs = (dist_f, angle_f_rel, dist_p, angle_p_rel)
            
            output = nets[i].activate(inputs)
            creature.move(output)
            
            # 2. 수명 감소 및 사망 판정
            creature.life -= 1 
            ge[i].fitness += 0.01 # [수정] 생존 보너스 대폭 감소
            
            # 포식자 충돌 사망 판정 (가장 가까운 포식자 P1과만 체크)
            if predator1:
                predator_distance = math.sqrt((creature.x - predator1.x)**2 + (creature.y - predator1.y)**2)
                if predator_distance < creature.rect.width / 2 + predator1.rad: 
                    creature.life = 0 
                    ge[i].fitness -= 20 
            
            if creature.life <= 0:
                creatures.pop(i)
                nets.pop(i)
                ge.pop(i)
                continue 

            # 3. 먹이를 먹었는지 판별 
            for j in reversed(range(len(foods))):
                food_item = foods[j]
                distance = math.sqrt((creature.x - food_item.x)**2 + (creature.y - food_item.y)**2)
                
                if distance < 20:
                    ge[i].fitness += 100 
                    creature.life += 600 
                    
                    foods.pop(j)
                    foods.append(Food())
                    
                    break
        
        # --- 그리기 및 정보 표시 ---
        screen.fill((0, 0, 0))
        for food_item in foods: 
            food_item.draw(screen)
        
        for predator in predators:
            predator.draw(screen)
            
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
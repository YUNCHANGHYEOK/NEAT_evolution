import pygame
import neat
import os
import random
import math

# --- 1. 전역 설정 ---
WIN_WIDTH = 800       # 게임 화면 너비
WIN_HEIGHT = 600      # 게임 화면 높이
GEN = 0               # 현재 세대 수
FPS = 60              # 초당 60프레임
MAX_GEN_TIME = 1800   # 한 세대 최대 시간 (약 30초)
FOOD_COUNT = 5        # 먹이 개수


# --------------------------------
# ▣ 종 ID에 기반한 색상 생성 함수
# --------------------------------
def get_color_from_id(id):
    """각 genome 또는 species ID에 따라 고유 색을 만들어주는 함수"""
    if id <= 0:
        return (0, 150, 0)

    random.seed(id)  # ID를 기반으로 랜덤 고정
    r = random.randint(50, 200)
    g = random.randint(50, 200)
    b = random.randint(50, 200)
    random.seed(None)  # 랜덤 시드 초기화

    return (r, g, b)


# --------------------------------
# ▣ 먹이 클래스
# --------------------------------
class Food:
    def __init__(self):
        # 화면 내 무작위 위치에 생성
        self.x = random.randint(50, WIN_WIDTH - 50)
        self.y = random.randint(50, WIN_HEIGHT - 50)
        self.rad = 10  # 먹이의 반지름

    def draw(self, win):
        # 빨간색 원으로 먹이 표시
        pygame.draw.circle(win, (255, 0, 0), (self.x, self.y), self.rad)


# --------------------------------
# ▣ 생명체(Agent, Genome 개체) 클래스
# --------------------------------
class Creature:
    def __init__(self, species_id=0):
        # 초기 위치: 화면 중앙
        self.x = WIN_WIDTH / 2
        self.y = WIN_HEIGHT / 2

        self.vel = 5                    # 이동 속도
        self.rect = pygame.Rect(self.x, self.y, 20, 20)  # 충돌 박스
        self.life = 600                 # 생명력(프레임 단위), 현재 600/60=10초

        # 종 ID / 고유 색상
        self.species_id = species_id
        self.base_color = get_color_from_id(species_id)

    # 생명체 이동 함수 (NEAT의 출력값으로 이동)
    def move(self, output):
        # output = [up, down, left, right] 형태
        if output[0] > 0.5:
            self.y -= self.vel
        if output[1] > 0.5:
            self.y += self.vel
        if output[2] > 0.5:
            self.x -= self.vel
        if output[3] > 0.5:
            self.x += self.vel

        # 화면 밖으로 나가지 못하도록 제한
        self.x = max(0, min(self.x, WIN_WIDTH - 20))
        self.y = max(0, min(self.y, WIN_HEIGHT - 20))

        # rect 업데이트
        self.rect.topleft = (self.x, self.y)

    # 생명체 그리기
    def draw(self, win, font):
        # 체력 비례 색상 어둡게 (체력 낮으면 색 진해짐)
        life_ratio = min(1.0, self.life / 600.0) # 최대 체력 600 기준, 현재 체력는 0~1 사이

        r = int(self.base_color[0] * life_ratio)
        g = int(self.base_color[1] * life_ratio)
        b = int(self.base_color[2] * life_ratio)

        # 최소 밝기 보정
        r = max(10, r)
        g = max(10, g)
        b = max(10, b)

        # 현재 색
        current_color = (r, g, b)

        # 생명체 몸체 그리기
        pygame.draw.rect(win, current_color, self.rect)

        # 체력 숫자를 중앙에 표시
        life_text = font.render(str(int(self.life)), True, (255, 255, 255))
        win.blit(life_text, life_text.get_rect(center=self.rect.center))


# --------------------------------
# ▣ genome 평가 함수 (NEAT 핵심)
# --------------------------------
def eval_genomes(genomes, config):
    """
    각 세대의 모든 genome을 평가하는 함수.
    NEAT 알고리즘에서 반드시 필요한 함수이며,
    p.run()에서 자동으로 호출됨.
    """

    global GEN
    GEN += 1   # 세대 증가

    # pygame 초기화
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    nets = []       # 신경망 리스트
    creatures = []  # 생명체 리스트
    ge = []         # genome 객체 리스트

    foods = [Food() for _ in range(FOOD_COUNT)]  # 먹이 생성

    # --- 각 genome을 creature와 연결 ---
    for genome_id, genome in genomes:
        genome.fitness = 0  # 초기 fitness

        # FeedForward 네트워크 구성
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

        # species_id가 genome에 있으면 가져오기
        species_id = getattr(genome, "species_id", 1)
        creatures.append(Creature(species_id))

        ge.append(genome)

    run = True
    total_time = 0  # 한 세대 진행 시간

    # 글꼴 설정
    info_font = pygame.font.SysFont("comicsans", 30)
    creature_font = pygame.font.SysFont("comicsans", 16)

    # ==============================
    # ▣ 메인 게임 루프 (세대 평가)
    # ==============================
    while run and len(creatures) > 0:
        total_time += 1

        # 제한 시간 초과 시 종료 → 다음 세대로
        if total_time > MAX_GEN_TIME:
            break

        # 종료 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # ——————————————
        # ▣ 모든 생명체 업데이트
        # ——————————————
        for i in reversed(range(len(creatures))):
            creature = creatures[i]

            # 가장 가까운 먹이 찾기
            closest_food = None
            min_distance = float("inf")

            for f in foods:
                distance = math.sqrt((creature.x - f.x)**2 + (creature.y - f.y)**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_food = f

            # 먹이 방향으로 입력값 구성 (dx, dy)
            if closest_food:
                dx = closest_food.x - creature.x
                dy = closest_food.y - creature.y
                output = nets[i].activate((dx, dy))
            else:
                # 먹이가 없다면 0 입력
                output = nets[i].activate((0, 0))

            # 생명체 이동
            creature.move(output)

            # 매 프레임마다 체력 감소
            creature.life -= 1
            if creature.life <= 0:
                # 죽으면 pop
                creatures.pop(i)
                nets.pop(i)
                ge.pop(i)
                continue

            # 기본 생존 가점
            ge[i].fitness += 0.1

            # 먹이 먹기
            if closest_food and min_distance < 20:
                ge[i].fitness += 20    # 큰 보상
                creature.life += 300   # 체력 회복

                foods.remove(closest_food)
                foods.append(Food())   # 새로운 먹이 추가

        # ——————————————
        # ▣ 화면 그리기
        # ——————————————
        screen.fill((0, 0, 0))

        # 먹이 그리기
        for food_item in foods:
            food_item.draw(screen)

        # 생명체 그리기
        for creature in creatures:
            creature.draw(screen, creature_font)

        # 정보창 표시
        remain_time = max(0, (MAX_GEN_TIME - total_time) // FPS)
        unique_species = len(set(c.species_id for c in creatures))

        info_text = info_font.render(
            f"Gen: {GEN} | Alive: {len(creatures)} | Species: {unique_species} | Time Left: {remain_time}s",
            True, (255, 255, 255)
        )
        screen.blit(info_text, (10, 10))

        pygame.display.update()
        clock.tick(FPS)


# --------------------------------
# ▣ NEAT 실행 함수
# --------------------------------
def run(config_path):
    # NEAT 설정 파일 읽기
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # NEAT population 생성
    p = neat.Population(config)

    # 기본 콘솔 출력 및 통계 reporter 추가
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # NEAT 실행 —> eval_genomes()를 50세대 동안 반복 호출
    winner = p.run(eval_genomes, 50)


# --------------------------------
# 메인
# --------------------------------
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)

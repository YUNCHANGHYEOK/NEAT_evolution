import pygame
import neat
import os
import random
import math

# --- 1. 전역 설정 ---
WIN_WIDTH = 800          # 윈도우 가로 크기
WIN_HEIGHT = 600         # 윈도우 세로 크기
GEN = 0                  # 현재 세대 수 (세대가 바뀔 때마다 +1)
FPS = 60                 # 초당 프레임 수 (60FPS)
MAX_GEN_TIME = 1800      # 한 세대당 최대 진행 프레임 수 (60FPS * 30초 = 1800)

# --------------------------------
# 먹이(빨간 원) 클래스
# --------------------------------
class Food:
    def __init__(self):
        # 화면 안의 랜덤 위치에 생성
        self.x = random.randint(50, WIN_WIDTH - 50)
        self.y = random.randint(50, WIN_HEIGHT - 50)
        self.rad = 10  # 반지름

    def draw(self, win):
        # 화면에 빨간색 원으로 먹이 그리기
        pygame.draw.circle(win, (255, 0, 0), (self.x, self.y), self.rad)

# --------------------------------
# 생명체(초록 사각형) 클래스
# --------------------------------
class Creature:
    def __init__(self):
        # 시작 위치: 화면 중앙
        self.x = WIN_WIDTH / 2
        self.y = WIN_HEIGHT / 2
        
        self.vel = 5  # 이동 속도 (한 번 움직일 때 몇 픽셀 이동할지)
        
        # pygame.Rect(x, y, width, height) 로 사각형 영역 생성
        self.rect = pygame.Rect(self.x, self.y, 20, 20)
        
        # 수명: 프레임 단위. 600프레임이면 10초(60FPS 기준)
        self.life = 600

    def move(self, output):
        """
        신경망의 출력값(output)에 따라 생명체를 상/하/좌/우로 이동시키는 함수.
        output: [o0, o1, o2, o3] 형태 (0~1 범위라고 가정)
          - o0 > 0.5: 위로 이동
          - o1 > 0.5: 아래로 이동
          - o2 > 0.5: 왼쪽으로 이동
          - o3 > 0.5: 오른쪽으로 이동
        """
        # 신경망 출력에 따른 방향 이동
        if output[0] > 0.5:
            self.y -= self.vel  # 위쪽( y 감소 )
        if output[1] > 0.5:
            self.y += self.vel  # 아래쪽( y 증가 )
        if output[2] > 0.5:
            self.x -= self.vel  # 왼쪽( x 감소 )
        if output[3] > 0.5:
            self.x += self.vel  # 오른쪽( x 증가 )

        # 화면 밖으로 나가지 못하게 좌표를 제한
        self.x = max(0, min(self.x, WIN_WIDTH - 20))
        self.y = max(0, min(self.y, WIN_HEIGHT - 20))

        # rect 위치도 갱신
        self.rect.topleft = (self.x, self.y)

    def draw(self, win, font):
        """
        생명체를 화면에 그리는 함수.
        - 초록색 사각형
        - 사각형 중앙에 현재 life(수명) 숫자를 실시간으로 표시
        """
        # life를 0~600 범위라고 보고, 0~1로 정규화 후 life_ratio 계산
        # 1.0을 넘지 않도록 min 사용 (색상값 255 초과 방지)
        life_ratio = min(1.0, self.life / 600.0)
        green_intensity = int(255 * life_ratio)  # 0 ~ 255 범위의 초록색 강도
        current_color = (0, green_intensity, 0)  # (R=0, G=green_intensity, B=0)

        # 생명체(녹색 사각형) 그리기
        pygame.draw.rect(win, current_color, self.rect)

        # --- 여기서 life 숫자 표시 ---
        # life를 정수로 변환해서 문자열로 만들기
        life_text_str = str(int(self.life))
        life_text_surf = font.render(life_text_str, True, (255, 255, 255))  # 흰색 문자
        life_text_rect = life_text_surf.get_rect(center=self.rect.center)   # 사각형 중앙에 정렬

        # 숫자를 화면에 블릿
        win.blit(life_text_surf, life_text_rect)

# --------------------------------
# NEAT가 각 genome(유전자 집합)을 평가하는 함수
# --------------------------------
def eval_genomes(genomes, config):
    """
    NEAT 라이브러리가 각 세대마다 호출하는 평가 함수.
    - genomes: 현재 세대의 (genome_id, genome) 튜플 리스트
    - config: NEAT 설정 객체
    이 함수 내부에서 pygame 시뮬레이션을 돌리며,
    각 genome에 대해 fitness(적합도)를 계산해 준다.
    """
    global GEN
    GEN += 1  # 세대 수 증가 (1세대, 2세대, ...)

    # pygame 초기화 및 화면 생성
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    # NEAT 신경망, 생명체, genome(유전자) 리스트
    nets = []       # 각 genome에 대응되는 신경망
    creatures = []  # 각 신경망을 제어하는 생명체 객체
    ge = []         # genome 객체 (fitness 기록용)

    # 첫 번째 먹이 생성
    food = Food()

    # --- genome마다 생명체와 신경망 생성 ---
    for genome_id, genome in genomes:
        genome.fitness = 0  # 초기 적합도 0으로 설정

        # genome + config로부터 feedforward 신경망 생성
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

        # 생명체 하나 생성해서 신경망과 매칭
        creatures.append(Creature())

        # genome도 리스트에 저장 (나중에 fitness 업데이트용)
        ge.append(genome)

    run = True
    total_time = 0  # 이번 세대가 진행된 총 프레임 수

    # 화면 상단 정보 표시용 폰트(조금 크게)
    info_font = pygame.font.SysFont("comicsans", 30)
    # 각 생명체 위에 life 숫자 표시용 폰트(조금 작게)
    creature_font = pygame.font.SysFont("comicsans", 16)

    # --- 메인 루프: 이 세대가 끝날 때까지 반복 ---
    while run and len(creatures) > 0:
        total_time += 1  # 프레임 카운트 증가

        # 하드 타임 리밋: MAX_GEN_TIME(30초) 이상 지나면 세대 종료
        if total_time > MAX_GEN_TIME:
            break

        # pygame 이벤트 처리 (창 닫기 등)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # --- 역순 반복: 리스트에서 pop 할 때 인덱스 꼬임 방지 ---
        for i in reversed(range(len(creatures))):
            creature = creatures[i]

            # 1. 입력값 계산: 먹이와 생명체 사이의 거리차 (dx, dy)
            dx = food.x - creature.x
            dy = food.y - creature.y

            # 2. 신경망에 입력 전달 → 출력 받기
            #   config에서 input 노드 수가 2, output 노드 수가 4가 되도록 설정되어 있어야 함
            output = nets[i].activate((dx, dy))

            # 생명체 이동
            creature.move(output)

            # 3. 수명 감소
            creature.life -= 1  # 매 프레임마다 life 1 감소

            # 수명이 다 된 생명체는 제거
            if creature.life <= 0:
                creatures.pop(i)
                nets.pop(i)
                ge.pop(i)
                continue  # 다음 생명체로 넘어감

            # 4. 생존 보너스: 살아 있는 동안 조금씩 fitness 증가
            ge[i].fitness += 0.1

            # 5. 먹이를 먹었는지 판정 (거리로 체크)
            distance = math.sqrt((creature.x - food.x) ** 2 + (creature.y - food.y) ** 2)
            if distance < 20:  # 생명체와 먹이가 일정 거리 이내면 "먹은 것"으로 처리
                ge[i].fitness += 20      # 먹으면 큰 보상
                creature.life += 300     # 수명 300프레임(약 5초) 증가
                food = Food()           # 새로운 위치에 먹이 생성

        # --- 화면 그리기 ---
        screen.fill((0, 0, 0))  # 배경을 검은색으로 초기화

        # 먹이 그리기
        food.draw(screen)

        # 각 생명체 그리기 (사각형 + life 숫자)
        for creature in creatures:
            creature.draw(screen, creature_font)

        # 상단 정보 텍스트 (세대, 살아있는 개체 수, 남은 시간)
        remain_time = max(0, (MAX_GEN_TIME - total_time) // FPS)  # 초 단위 남은 시간
        info_text = info_font.render(
            f"Gen: {GEN} | Alive: {len(creatures)} | Time Left: {remain_time}s",
            True,
            (255, 255, 255)
        )
        screen.blit(info_text, (10, 10))

        # 화면 업데이트
        pygame.display.update()
        # FPS 고정
        clock.tick(FPS)

# --------------------------------
# NEAT 실행 함수
# --------------------------------
def run(config_path):
    """
    NEAT 설정 파일을 읽어서
    - Population(집단) 생성
    - Reporter(로그 출력용) 등록
    - p.run(...) 으로 여러 세대를 돌리는 함수
    """
    # NEAT 설정 로드
    config = neat.config.Config(
        neat.DefaultGenome,        # genome(유전자) 기본 타입
        neat.DefaultReproduction,  # 번식 방식
        neat.DefaultSpeciesSet,    # 종 분류 방식
        neat.DefaultStagnation,    # 정체(stagnation) 처리 방식
        config_path                # config 파일 경로
    )

    # 개체 집단(Population) 생성
    p = neat.Population(config)

    # 콘솔에 학습 진행 상황을 출력하는 Reporter 추가
    p.add_reporter(neat.StdOutReporter(True))
    # 통계 수집용 Reporter 추가
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # p.run(평가함수, 세대 수)
    # → eval_genomes 함수를 최대 50세대까지 실행
    winner = p.run(eval_genomes, 50)

    # winner는 최종적으로 가장 높은 fitness를 가진 genome
    # 여기서는 따로 사용하지 않았지만,
    # 나중에 저장하거나 재사용할 수 있음
    # print("Best genome:", winner)

# --------------------------------
# 메인 실행 부분
# --------------------------------
if __name__ == "__main__":
    # 현재 파일이 위치한 디렉토리 경로
    local_dir = os.path.dirname(__file__)
    # config-feedforward.txt 파일의 전체 경로 생성
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    # NEAT 실행
    run(config_path)

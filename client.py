import pygame
import random

# Pygame 초기화
pygame.init()

# 화면 크기 설정
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Random Shape Placement")

# 랜덤한 색상 리스트
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

# 도형의 모양 리스트
SHAPES = ['circle', 'rectangle', 'triangle']

# 도형 클래스
class Shape:
    def __init__(self, shape_type, color, x, y):
        self.shape_type = shape_type
        self.color = color
        self.x = x
        self.y = y

    def draw(self):
        if self.shape_type == 'circle':
            pygame.draw.circle(screen, self.color, (self.x, self.y), 40)
        elif self.shape_type == 'rectangle':
            pygame.draw.rect(screen, self.color, (self.x - 40, self.y - 40, 80, 80))
        elif self.shape_type == 'triangle':
            pygame.draw.polygon(screen, self.color, [(self.x, self.y - 40), (self.x - 40, self.y + 40), (self.x + 40, self.y + 40)])

# 3x3 그리드에 도형 생성
shapes = []
for i in range(3):
    for j in range(3):
        shape_type = random.choice(SHAPES)
        color = random.choice(COLORS)
        x = (SCREEN_WIDTH // 3) * (i + 1)
        y = (SCREEN_HEIGHT // 3) * (j + 1)
        shape = Shape(shape_type, color, x, y)
        shapes.append(shape)

# 가운데에 일치하는 도형 선택
matching_shape = random.choice(shapes)

# 가운데에 도형 배치
matching_shape.x = SCREEN_WIDTH // 2
matching_shape.y = SCREEN_HEIGHT // 2

# 게임 루프
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 배경 색상 설정
    screen.fill((0, 0, 0))

    # 도형 그리기
    for shape in shapes:
        shape.draw()

    pygame.display.flip()
import time
import cv2
import sys
import pygame
import mediapipe
import random
import copy
import pyautogui
import numpy as np

# Pygame 초기화
pygame.init()

# 화면 크기 설정
SCREEN_WIDTH = pyautogui.size().width
SCREEN_HEIGHT = pyautogui.size().height
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Random Shape Placement")

face_mesh_landmarks = mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cam = cv2.VideoCapture(0)
# 카메라 프레임 크기 설정
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
screen_w, screen_h = pyautogui.size()
mouse_x, mouse_y = screen_w // 2, screen_h // 2

yes_Img = pygame.image.load("yes_Btn.png")
no_Img = pygame.image.load("No_Btn.png")
title_Img = pygame.image.load("title.png")

start_ticks = pygame.time.get_ticks()
# Text
font = pygame.font.SysFont("arial", 30, True, True)

# 랜덤한 색상 리스트
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (125, 125, 0)]

# 도형의 모양 리스트
SHAPES = ['circle', 'rectangle', 'triangle']

level_index = 0
eye_cnt = 0
cnt = 0


class Button:
    def __init__(self, img_in, x, y, width, height, img_act, x_act, y_act, action=None):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if x + width > mouse[0] > x and y + height > mouse[1] > y:
            screen.blit(img_act, (x_act, y_act))
            if click[0] and action != None:
                action()
        else:
            screen.blit(img_in, (x, y))


class Shape:
    def __init__(self, shape_type, color, x, y):
        self.shape_type = shape_type
        self.color = color
        self.x = x
        self.y = y

    def draw(self):
        if self.shape_type == 'circle':
            pygame.draw.circle(screen, self.color, (self.x, self.y), 80)
        elif self.shape_type == 'rectangle':
            pygame.draw.rect(screen, self.color, (self.x - 80, self.y - 80, 160, 160))
        elif self.shape_type == 'triangle':
            pygame.draw.polygon(screen, self.color,
                                [(self.x, self.y - 80), (self.x - 80, self.y + 80), (self.x + 80, self.y + 80)])


def quitgame():
    pygame.quit()
    sys.exit()


def level():
    global level_index
    level_index += 1


shapes = []
for i in range(3):
    for j in range(3):
        shape_type = random.choice(SHAPES)
        color = random.choice(COLORS)
        if ((i == 1) and (j == 1)):
            pass
        else:
            x = (SCREEN_WIDTH // 3) * (i + 0.5)
            y = (SCREEN_HEIGHT // 3) * (j + 0.5)
            shape = Shape(shape_type, color, x, y)
            shapes.append(shape)

# 가운데에 일치하는 도형 선택
matching_shape = copy.deepcopy(random.choice(shapes))

# 가운데에 도형 배치
matching_shape.x = SCREEN_WIDTH // 2
matching_shape.y = SCREEN_HEIGHT // 2

# 게임 루프
# 마우스 클릭 이벤트 처리
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            for shape in shapes:
                if shape != matching_shape and shape.x - 40 <= pos[0] <= shape.x + 40 and shape.y - 40 <= pos[
                    1] <= shape.y + 40:
                    if shape.shape_type == matching_shape.shape_type and shape.color == matching_shape.color:
                        # 게임을 재시작
                        shapes = []
                        for i in range(3):
                            for j in range(3):
                                if (i == 1) and (j == 1):
                                    pass
                                else:
                                    x = (SCREEN_WIDTH // 3) * (i + 0.5)
                                    y = (SCREEN_HEIGHT // 3) * (j + 0.5)
                                    while True:
                                        shape_type = random.choice(SHAPES)
                                        color = random.choice(COLORS)
                                        if (shape_type, color) != (matching_shape.shape_type, matching_shape.color):
                                            break
                                    shape = Shape(shape_type, color, x, y)
                                    shapes.append(shape)
                        matching_shape = copy.deepcopy(random.choice(shapes))
                        matching_shape.x = SCREEN_WIDTH // 2
                        matching_shape.y = SCREEN_HEIGHT // 2
                        cnt += 1
                        if (cnt >= 2):
                            level()
                    else:
                        # 다른 도형을 클릭한 경우 아무 일도 하지 않음
                        pass
    # 배경 색상 설정
    screen.fill((48, 48, 56))
    if level_index == 0:
        for shape in shapes:
            shape.draw()
        matching_shape.draw()
        _, image = cam.read()
        image = cv2.flip(image, 1)
        window_h, window_w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = face_mesh_landmarks.process(rgb_image)
        all_face_landmark_points = processed_image.multi_face_landmarks
        if all_face_landmark_points:
            one_face_landmark_points = all_face_landmark_points[0].landmark
            for id, landmark_point in enumerate(one_face_landmark_points[474:478]):
                x = int(landmark_point.x * window_w)
                y = int(landmark_point.y * window_h)
                if id == 1:
                    mouse_x = int(screen_w / window_w * x)
                    mouse_y = int(screen_h / window_h * y)
                    pyautogui.moveTo(mouse_x, mouse_y)
                cv2.circle(image, (x, y), 3, (0, 0, 255))
            left_eye = [one_face_landmark_points[145], one_face_landmark_points[155]]
            for landmark_point in left_eye:
                x = int(landmark_point.x * window_w)
                y = int(landmark_point.y * window_h)

                cv2.circle(image, (x, y), 3, (0, 255, 255))
    elif level_index == 1:
        title = screen.blit(title_Img, ((SCREEN_WIDTH / 5) * 2, 150))
        # 버튼 생성
        Yes_Btn = Button(yes_Img, (SCREEN_WIDTH / 5) * 2.3, 260, 60, 20, yes_Img, (SCREEN_WIDTH / 5) * 2.3, 258, level)
        No_Btn = Button(no_Img, (SCREEN_WIDTH / 5) * 2.6, 260, 60, 20, no_Img, (SCREEN_WIDTH / 5) * 2.6, 258, quitgame)
    elif level_index == 2:
        # 카메라에서 프레임 읽기
        _, frame = cam.read()
        # 프레임 회전
        frame = cv2.flip(frame, 1)
        # 얼굴 랜드마크 검출
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_landmarks.process(rgb_frame)

        # 왼쪽 눈과 오른쪽 눈의 좌표 획득
        left_eye_landmarks = [145, 155]  # 왼쪽 눈 랜드마크 인덱스
        right_eye_landmarks = [374, 384]  # 오른쪽 눈 랜드마크 인덱스

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 왼쪽 눈 좌표 계산 및 표시
                    for landmark_index in left_eye_landmarks:
                        x = int(face_landmarks.landmark[landmark_index].x * frame.shape[1])
                        y = int(face_landmarks.landmark[landmark_index].y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

                    # 오른쪽 눈 좌표 계산 및 표시
                    for landmark_index in right_eye_landmarks:
                        x = int(face_landmarks.landmark[landmark_index].x * frame.shape[1])
                        y = int(face_landmarks.landmark[landmark_index].y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

                    # 왼쪽 눈과 오른쪽 눈의 y 좌표 차이 계산
                    left_eye_y_diff = abs(face_landmarks.landmark[left_eye_landmarks[0]].y -
                                          face_landmarks.landmark[left_eye_landmarks[1]].y)
                    right_eye_y_diff = abs(face_landmarks.landmark[right_eye_landmarks[0]].y -
                                           face_landmarks.landmark[right_eye_landmarks[1]].y)

                    if left_eye_y_diff < 0.01 and right_eye_y_diff < 0.01:  # 양쪽 눈을 감았을 때
                        eye_cnt += 1

        # # 빨간색 원으로 얼굴 랜드마크 표시
        # if results.multi_face_landmarks:
        #     for face_landmarks in results.multi_face_landmarks:
        #         for landmark in face_landmarks.landmark:
        #             x = int(landmark.x * frame.shape[1])
        #             y = int(landmark.y * frame.shape[0])
        #             cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # OpenCV 이미지를 Pygame 화면에 표시
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (SCREEN_WIDTH // 2 - 320, SCREEN_HEIGHT // 2))

        # 타이머 제작
        left_time = 30 - (pygame.time.get_ticks() - start_ticks) / 1000
        timer_text = font.render("Time Left : " + str(left_time), True, (255, 255, 255))
        screen.blit(timer_text, ((SCREEN_WIDTH / 5) * 2, 100))
        if (left_time <= 0):
            level()

        # 깜빡임 횟수 표시
        text = font.render("Number of blinks : " + str(eye_cnt), True, (255, 255, 255))
        screen.blit(text, ((SCREEN_WIDTH / 5) * 2, 150))

    elif level_index == 3:
        text = font.render("Number of blinks : " + str(eye_cnt), True, (255, 255, 255))
        screen.blit(text, ((SCREEN_WIDTH / 5) * 2, 100))
        # 1분 20회 평균

    pygame.display.flip()
cam.release()
cv2.destroyAllWindows()
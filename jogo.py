import cv2
import mediapipe as mp

# importa o video
video = cv2.VideoCapture('pedra-papel-tesoura.mp4')
# permite desenhar formas e traçados sobre imagens e vídeos
mp_drawing = mp.solutions.drawing_utils
# contém estilos predefinidos para desenhar forma e traços
mp_drawing_styles = mp.solutions.drawing_styles
# detecta mãos e características associadas em imagens e vídeos
mp_hands = mp.solutions.hands


# função que identifica o gesto da mão
def getHandGesture(hand_landmarks):
    landmarks = hand_landmarks.landmark

    # calcula a distância entre os dedos
    dist1 = ((landmarks[8].x - landmarks[12].x)**2 +
             (landmarks[8].y - landmarks[12].y)**2)**0.5
    dist2 = ((landmarks[8].x - landmarks[4].x)**2 +
             (landmarks[8].y - landmarks[4].y)**2)**0.5

    # verifica se o gesto corresponde a pedra, papel ou tesoura
    if dist1 < 0.04 and dist2 < 0.04:
        return "pedra"
    elif dist1 > 0.06 and dist2 > 0.06:
        return "tesoura"
    else:
        return "papel"


# desenhando os pontos e traçados nas mãos
def drawHandStrokes(mhl):
    for hand_landmarks in mhl:
        mp_drawing.draw_landmarks(
            img,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )


# identifica as maos do primeiro e do segundo jogador
def detectPlayerHands(mhl):
    hand_one, hand_two = mhl
    # menor valor de X da primeira mao detectada
    min_x_hand_1 = min(list(
        map(lambda l: l.x, hand_one.landmark)))
    # menor valor de X da segunda mao detectada
    min_x_hand_2 = min(list(
        map(lambda l: l.x, hand_two.landmark)))

    # a primeira mão é a que inicia na menor posição de X na tela
    first_player_hand = hand_one if min_x_hand_1 < min_x_hand_2 else hand_two
    second_player_hand = hand_two if min_x_hand_1 < min_x_hand_2 else hand_one

    return first_player_hand, second_player_hand


# define o jogador vencedor
def defineWinner(gesture_one, gesture_two):
    if gesture_one == gesture_two:
        return 0
    elif gesture_one == "papel":
        return 1 if gesture_two == "pedra" else 2
    elif gesture_one == "pedra":
        return 1 if gesture_two == "tesoura" else 2
    elif gesture_one == "tesoura":
        return 1 if gesture_two == "papel" else 2


first_player_gesture = None
second_player_gesture = None
winning_player = None  # número do jogador que venceu o round
scores = [0, 0]

# instancia do objeto que permite a detecção de mãos a partir de um fluxo de imagens
hands = mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    success, img = video.read()

    # se nao for possivel ler o frame (normalmente quando o video termina), o loop é encerrado
    if not success:
        break

    # recebe uma imagem e retorna informações como a posição e orientação das mãos
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # lista dos pontos de referencia das mãos detectadas
    mhl = results.multi_hand_landmarks

    # se nenhuma mao for detectada ou
    # mais ou menos de duas maos forem detectadas
    # a interação do frame no loop é pulada
    if not mhl or len(mhl) != 2:
        continue

    drawHandStrokes(mhl)

    first_player_hand, second_player_hand = detectPlayerHands(mhl)

    # verifica quem venceu se um dos gestos mudarem em relação ao gesto detectado anteriormente no outro frame do video
    new_fpg = getHandGesture(first_player_hand)
    new_spg = getHandGesture(second_player_hand)

    if (new_fpg != first_player_gesture or new_spg != second_player_gesture):
        # pega o gesto da mao da esquerda
        first_player_gesture = new_fpg
        # pega o gesto da mao da direita
        second_player_gesture = new_spg

        winning_player = defineWinner(
            first_player_gesture, second_player_gesture)

        if winning_player == 1:
            scores[0] += 1
        elif winning_player == 2:
            scores[1] += 1

        round_result = "Empate!" if winning_player == 0 else f"Jogador {winning_player} venceu!"
        print(f"{first_player_gesture} x {second_player_gesture} = {round_result}")

    # exibindo os textos na tela
    font = cv2.FONT_HERSHEY_SIMPLEX

    # exibindo o placar no centro superior da tela
    score_text = f"{scores[0]} x {scores[1]}"
    score_size, _ = cv2.getTextSize(score_text, font, 2, 5)
    cv2.putText(img, score_text, [(img.shape[1] - score_size[0]) // 2, 100], font,
                2, (50, 50, 50), 5)

    # exibindo o resultado da rodada no centro inferior da tela
    result_size, _ = cv2.getTextSize(round_result, font, 2, 2)
    cv2.putText(img, round_result, [(img.shape[1] - result_size[0]) // 2, img.shape[0] - result_size[1]], font,
                2, (70, 70, 70), 2)

    # exibindo as jogadas do primeiro jogador no centro esquerdo da tela
    first_player_text = "Jogador 1"
    first_player_size, _ = cv2.getTextSize(first_player_text, font, 1.2, 2)
    cv2.putText(img, first_player_text, (50, img.shape[0] // 2 - first_player_size[1]),
                font, 1.2, (255, 0, 0), 2)
    cv2.putText(img, first_player_gesture, (50, img.shape[0] // 2 - first_player_size[1] + 70),
                font, 2, (100, 100, 100), 2)

    # exibindo as jogadas do segundo jogador no centro direito da tela
    second_player_text = "Jogador 2"
    second_player_size, _ = cv2.getTextSize(
        second_player_text, font, 2, 2)
    cv2.putText(img, second_player_text, (img.shape[1] - second_player_size[0], img.shape[0] // 2 - second_player_size[1]),
                font, 1.2, (0, 0, 255), 2)
    cv2.putText(img, second_player_gesture, (img.shape[1] - second_player_size[0], img.shape[0] // 2 - second_player_size[1] + 70),
                font, 2, (100, 100, 100), 2)

    # cria uma janela e define o tamanho
    cv2.namedWindow('Pedra, papel ou tesoura', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pedra, papel ou tesoura', 960, 540)
    cv2.imshow('Pedra, papel ou tesoura', img)
    cv2.waitKey(10)

video.release()
cv2.destroyAllWindows()

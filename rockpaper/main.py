import cv2
import time
from pathlib import Path
from ultralytics import YOLO

model_path = "best.pt"
model = YOLO(model_path)

window_name = "YOLO"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)
player_1_hand = ""
player_2_hand = ""
state = "idle"
prev_time = 0
timer = 5
result_display_time = 2
result_time = 0


def find_winner():
    outcomes = {
        ("rock", "scissors"): "player_1_won",
        ("scissors", "paper"): "player_1_won",
        ("paper", "rock"): "player_1_won",
        ("scissors", "rock"): "player_2_won",
        ("paper", "scissors"): "player_2_won",
        ("rock", "paper"): "player_2_won"
    }

    if player_1_hand == player_2_hand:
        return "draw"

    return outcomes.get((player_1_hand, player_2_hand), "idk")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Обработка результатов
    if results:
        boxes = results[0].boxes.xyxy
        if len(boxes) >= 2:
            labels = []
            for i in range(2):
                x1, y1, x2, y2 = boxes[i].numpy().astype("int")
                index = results[0].boxes.cls[i].item()
                name = results[0].names[index]
                labels.append(name.lower())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, f"{name}", (x1 + 20, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            player_1_hand, player_2_hand = labels
            if state == "idle":
                state = "wait"
                prev_time = time.time()
    if state == "wait":
        elapsed_time = round(time.time() - prev_time, 1)
        remaining_time = timer - elapsed_time
        if remaining_time <= 0:
            state = "result"
            game_result = find_winner()
            remaining_time = 0
            result_time = time.time()
        cv2.putText(frame, f"Time: {remaining_time}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if state == "result":
        cv2.putText(frame, f"{player_1_hand}, {player_2_hand}, Result: {game_result}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if time.time() - result_time >= result_display_time:
            state = "idle"

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

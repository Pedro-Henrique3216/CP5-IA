from ultralytics import YOLO
import cv2
import mediapipe as mp

import os

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "data", "teste.jpg")

model = YOLO("yolov8n.pt") 
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=True)


if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(
        f"Imagem não encontrada: {IMAGE_PATH}\n" \
        "Coloque a imagem de teste em visao-computacional/data/teste.jpg ou altere IMAGE_PATH"
    )

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise RuntimeError(f"Falha ao ler a imagem (arquivo presente mas cv2 retornou None): {IMAGE_PATH}")

results = model(image, verbose=False)[0]

for box in results.boxes:
    cls = int(box.cls[0])
    if cls != 0:
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    crop = image[y1:y2, x1:x2]

    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_res = holistic.process(rgb_crop)

    gesture = "Normal"

    if mp_res.right_hand_landmarks or mp_res.left_hand_landmarks:
        for hand in [mp_res.right_hand_landmarks, mp_res.left_hand_landmarks]:
            if hand:
                wrist = hand.landmark[0]
                index = hand.landmark[8]
                if index.y < wrist.y:
                    gesture = "Mão Erguida"
                    break

    cv2.rectangle(image, (x1, y1), (x2, y2),
                  (0, 255, 0) if gesture == "Mão Erguida" else (255, 255, 255),
                  2)
    cv2.putText(image, gesture, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if gesture == "Mão Erguida" else (255, 255, 255),
                2)

cv2.imwrite("output_result.jpg", image)
cv2.imshow("Resultado", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

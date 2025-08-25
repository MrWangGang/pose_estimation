from ultralytics import YOLO
import cv2
import numpy as np
import math

# 加载模型
model = YOLO('./model/best.pt')  # 替换为你的模型路径

# 打开视频文件
video_path = './test_data/video/fall.mp4'  # 替换为你的视频路径
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件，请检查路径。")
    exit()

# 关键点中文名称映射（COCO顺序）
JOINT_NAMES = [
    "鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩",
    "左肘", "右肘", "左腕", "右腕", "左髋", "右髋",
    "左膝", "右膝", "左踝", "右踝"
]

def calculate_center_point(keypoints, point_indices):
    total_x, total_y, count = 0, 0, 0
    for index in point_indices:
        if keypoints[index][0] > 0 and keypoints[index][1] > 0:
            total_x += keypoints[index][0]
            total_y += keypoints[index][1]
            count += 1
    return (total_x / count, total_y / count) if count > 0 else None

# 骨架连接顺序（COCO）
skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("视频读取结束")
        break

    if frame is not None:
        results = model(frame)
        for result_index, result in enumerate(results):
            keypoints = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)

            for i, person_keypoints in enumerate(keypoints):
                if person_keypoints.size == 0:
                    continue

                print(f"\n--- 第 {i + 1} 个人的关键点信息 ---")
                for joint_index in range(len(person_keypoints)):
                    x, y = person_keypoints[joint_index]
                    conf = keypoints_conf[i][joint_index]
                    joint_name = JOINT_NAMES[joint_index]
                    print(f"{joint_name:<4}：坐标 ({x:.2f}, {y:.2f})，置信度 {conf:.2f}")

                # 检查左髋和右髋关键点是否都未检测到ﬁ
                left_hip_x, left_hip_y = person_keypoints[11]
                right_hip_x, right_hip_y = person_keypoints[12]
                if left_hip_x <= 0 and left_hip_y <= 0 and right_hip_x <= 0 and right_hip_y <= 0:
                    fall = False
                    angle = 0
                    aspect_ratio = 0
                    cy = 0
                    wy = 0
                else:
                    # 计算角度
                    chest_point = calculate_center_point(person_keypoints, [5, 6])
                    waist_point = calculate_center_point(person_keypoints, [11, 12])
                    if chest_point and waist_point:
                        cx, cy = chest_point
                        wx, wy = waist_point
                        v1 = (0, wy - cy)
                        v2 = (wx - cx, wy - cy)
                        dot = v1[0] * v2[0] + v1[1] * v2[1]
                        norm1 = math.hypot(*v1)
                        norm2 = math.hypot(*v2)
                        angle = math.degrees(math.acos(dot / (norm1 * norm2))) if norm1 > 0 and norm2 > 0 else 0
                    else:
                        angle = 0
                        cy, wy = 0, 0

                    # 提取耳朵和胯部关键点
                    ear_and_hip_indices = [3, 4, 11, 12]
                    valid_points = [person_keypoints[idx] for idx in ear_and_hip_indices if person_keypoints[idx][0] > 0 and person_keypoints[idx][1] > 0]
                    if valid_points:
                        x_coords = [p[0] for p in valid_points]
                        y_coords = [p[1] for p in valid_points]
                        width = max(x_coords) - min(x_coords)
                        height = max(y_coords) - min(y_coords)
                        aspect_ratio = width / height if height > 0 else 0
                    else:
                        aspect_ratio = 0

                    # 判断是否跌倒
                    fall = angle > 60 or cy > wy or aspect_ratio > 3 / 5

                print(f"[调试信息] 角度: {angle:.2f}°，宽高比: {aspect_ratio:.2f}，胸Y: {cy:.2f}，腰Y: {wy:.2f}")

                # 可视化
                x1, y1, x2, y2 = boxes[i]
                color = (0, 0, 255) if fall else (0, 255, 255)
                text = "Fallen" if fall else ""
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # 画关键点
                for point in person_keypoints:
                    x, y = int(point[0]), int(point[1])
                    if x > 0 and y > 0:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # 连线骨架
                for connection in skeleton:
                    part_a, part_b = connection[0] - 1, connection[1] - 1
                    if part_a < len(person_keypoints) and part_b < len(person_keypoints):
                        if person_keypoints[part_a].any() and person_keypoints[part_b].any():
                            x1, y1 = int(person_keypoints[part_a][0]), int(person_keypoints[part_a][1])
                            x2, y2 = int(person_keypoints[part_b][0]), int(person_keypoints[part_b][1])
                            cv2.line(frame, (x1, y1), (x2, y2), color, 1)

        cv2.imshow('Pose Detection', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
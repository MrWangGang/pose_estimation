import streamlit as st
import cv2
import numpy as np
import math
from ultralytics import YOLO
import io
import tempfile


# 加载模型
model = YOLO('./model/best.pt')  # 替换为你的模型路径
USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "pass123"
}
# 关键点中文名称映射（COCO顺序）
JOINT_NAMES = [
    "鼻子", "左眼", "右眼", "左耳", "右耳", "左肩", "右肩",
    "左肘", "右肘", "左腕", "右腕", "左髋", "右髋",
    "左膝", "右膝", "左踝", "右踝"
]

# 骨架连接顺序（COCO）
skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]

def calculate_center_point(keypoints, point_indices):
    total_x, total_y, count = 0, 0, 0
    for index in point_indices:
        if keypoints[index][0] > 0 and keypoints[index][1] > 0:
            total_x += keypoints[index][0]
            total_y += keypoints[index][1]
            count += 1
    return (total_x / count, total_y / count) if count > 0 else None

def predict_pose(frame):
    results = model(frame)  # 使用YOLO模型预测
    for result_index, result in enumerate(results):
        # 输出预测结果，帮助调试

        # 检查 keypoints 是否有效
        if result.keypoints is None:
            st.warning(f"未检测到关键点 (帧 {result_index})")
            continue  # 跳过当前结果

        try:
            # 如果 keypoints 不是 None，继续处理
            keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
            keypoints_conf = result.keypoints.conf.cpu().numpy() if result.keypoints is not None else None
            boxes = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes is not None else None

            # 检查是否得到有效的关键点和边界框
            if keypoints is None or boxes is None:
                st.warning(f"没有检测到有效的关键点或边界框 (帧 {result_index})")
                continue  # 如果没有关键点或边界框，跳过

            for i, person_keypoints in enumerate(keypoints):
                if person_keypoints.size == 0:
                    continue  # 如果关键点为空，跳过

                # 计算跌倒检测相关的逻辑
                left_hip_x, left_hip_y = person_keypoints[11]
                right_hip_x, right_hip_y = person_keypoints[12]
                if left_hip_x <= 0 and left_hip_y <= 0 and right_hip_x <= 0 and right_hip_y <= 0:
                    fall = False
                    angle = 0
                    aspect_ratio = 0
                    cy = 0
                    wy = 0
                else:
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

                    fall = angle > 60 or cy > wy or aspect_ratio > 3 / 5

                # 可视化
                x1, y1, x2, y2 = boxes[i]
                color = (0, 0, 255) if fall else (0, 255, 255)
                text = "Fallen" if fall else ""
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                for point in person_keypoints:
                    x, y = int(point[0]), int(point[1])
                    if x > 0 and y > 0:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                for connection in skeleton:
                    part_a, part_b = connection[0] - 1, connection[1] - 1
                    if part_a < len(person_keypoints) and part_b < len(person_keypoints):
                        if person_keypoints[part_a].any() and person_keypoints[part_b].any():
                            x1, y1 = int(person_keypoints[part_a][0]), int(person_keypoints[part_a][1])
                            x2, y2 = int(person_keypoints[part_b][0]), int(person_keypoints[part_b][1])
                            cv2.line(frame, (x1, y1), (x2, y2), color, 1)

        except Exception as e:
            continue  # 捕获异常并跳过当前帧

    return frame






# Streamlit 页面逻辑
if "logged_in" in st.session_state and st.session_state["logged_in"]:
    page = st.session_state.get("page", "首页")
else:
    page = "登录"
# 登录验证函数
def login(username, password):
    # 验证用户名和密码
    return USER_CREDENTIALS.get(username) == password

# 登录页面配置
if page == "登录":
    st.set_page_config(page_title="登录页面", layout="centered")
    st.markdown("<h1 style='text-align: center;'>🔐 登录</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("用户名", key="username_input")
            password = st.text_input("密码", type="password", key="password_input")
            submitted = st.form_submit_button("登录")
            if submitted:
                input_username = st.session_state["username_input"]
                input_password = st.session_state["password_input"]
                if login(input_username, input_password):
                    st.success(f"欢迎，{input_username}！登录成功 🎉")
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = input_username
                    st.session_state["page"] = "首页"
                    if "rerun" not in st.session_state:
                        st.session_state["rerun"] = True
                        st.rerun()
                else:
                    st.error("用户名或密码错误，请重试。")

# 主页面内容
if page != "登录":
    # 左侧菜单栏
    menu = st.sidebar.radio("选择操作", ["首页", "图片预估", "视频预估"])

    if menu == "首页":
        st.title("欢迎来到姿态预估应用")
        st.write("选择左侧菜单栏进行操作。")

    # 图片预估页面
    elif menu == "图片预估":
        st.title("图片预估")
        uploaded_image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_image is not None:
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
            result_image = predict_pose(frame)
            st.image(result_image, channels="BGR", caption="检测结果", use_container_width=True)

    # 视频处理逻辑
    elif menu == "视频预估":
        st.title("视频预估")
        uploaded_video = st.file_uploader("上传视频", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_video is not None:
            # 将上传的视频文件转换为 BytesIO 对象
            video_bytes = uploaded_video.read()

            # 使用 tempfile 创建一个临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(video_bytes)
                temp_video_path = temp_video_file.name
                st.write(f"临时文件保存位置: {temp_video_path}")


        # 使用 OpenCV 打开临时文件
            cap = cv2.VideoCapture(temp_video_path)

            if not cap.isOpened():
                st.error("无法读取视频文件，请检查文件格式。")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率

                # 创建一个流式显示视频框架
                video_container = st.empty()

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
                slider_value = st.slider("视频播放进度", 0, frame_count, 0, 1)  # 使用滑块控制视频进度

                # 设置视频进度
                cap.set(cv2.CAP_PROP_POS_FRAMES,slider_value)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 逐帧计算姿态
                    result_frame = predict_pose(frame)

                    # 转换为适合 Streamlit 展示的格式
                    frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

                    # 使用 st.image 动态展示每一帧
                    video_container.image(frame_rgb, channels="RGB", use_container_width=True)

                cap.release()
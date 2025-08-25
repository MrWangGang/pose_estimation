from ultralytics import YOLO
import os


# 训练模型
def train_model():
    model = YOLO('./model/yolo11n-pose.pt')
    results = model.train(
        data='./datasets/datasets.yaml',
        epochs=100,
        imgsz=640,
        batch=128,  # 增加 batch size，提高稳定性
        workers=8,  # 使用 8 线程加载数据
        optimizer='AdamW',  # 使用 AdamW 优化器
        lr0=1e-5,  # 适当调整初始学习率
        lrf=0.01,  # 适当降低最终学习率
        momentum=0.937,  # 提高动量，加速收敛
        weight_decay=1e-4,  # 降低权重衰减，提高召回率
        degrees=5,  # 轻微旋转数据增强
        translate=0.1,  # 轻微平移数据增强
        scale=0.15,  # 适当缩放
        shear=2,  # 适当剪切
        fliplr=0.2,  # 适当水平翻转
        flipud=0.1,  # 适当垂直翻转
        pretrained=True,  # 使用预训练权重
        patience=30,  # 增加耐心值，防止过早停止
        cos_lr=True,  # 采用余弦退火学习率
        save_period=5,  # 更频繁地保存模型
        amp=True,  # 关闭/启用自动混合精度加速训练
    )
    return model


# 主程序
if __name__ == "__main__":
    # 训练模型
    trained_model = train_model()
    print("模型训练并保存完成。")
    
import time
from pathlib import Path
from ultralytics import YOLO
from video_precessing import predict_video
from utils import read_config


if __name__ == "__main__":
    # Чтение файла конфигурации
    config_path = Path("configs/config.yaml")
    config = read_config(config_path)

    # Создание модели
    model_version = Path(config.pop("model"))
    weights_path = Path(config.pop("weights_path"))
    model = YOLO(weights_path) if weights_path else YOLO(model_version)
    time_start = time.time()
    # Обработка видео
    flag = predict_video(model, **config)
    time_stop = time.time()
    if flag:
        print("Program ran successfully")
        print("Execution time: ", time_stop-time_start)
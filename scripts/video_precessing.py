import cv2
from typing import Any


def predict_video(
        model: Any,
        video_path: str,
        save_path: str,
        iou: float = 0.7,
        conf: float = 0.5,
        boxes: bool = True,
        labels: bool = True,
        line_width: int = 2,
        font_size: float = 0.8) -> bool:
    """
    Функция для детектирования людей на видео.

    Параметры
    ----------
    model: Any
        Модель для детекции
    video_path: str
        Путь к видео для обработки
    save_path: str
        Путь для сохранения обработанного видео
    iou: float
        Пороговое значение метрики IoU для учета предсказания модели
        Предсказания, имеющие значения IoU меньше порогового, не рассматриваются
    conf: float
        Порог степени уверенности модели в предсказании для отнесения к предсказываемому классу
    boxes: bool
        Флаг для отображения bounding box на видео
    labels: bool
        Флаг для отображения названий классов на видео
    line_width: int
        Ширина линии bounding box
    font_size: float
        Размер шрифта для отображения названия класса
    """
    caption = cv2.VideoCapture(video_path)
    assert caption.isOpened(), "Video is not opened"

    # Получение параметров исходного видео
    width = int(caption.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caption.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caption.get(cv2.CAP_PROP_FPS)

    # Создание VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    assert video_writer.isOpened(), "Failed to create video writer"

    while True:
        ret, frame = caption.read()
        if not ret:
            break

        # Обработка кадра
        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            verbose=False,
            classes=[0]
        )
        annotated_frame = results[0].plot(
            boxes=boxes,
            labels=labels,
            conf=conf,
            line_width=line_width,
            font_size=font_size
        )

        # Запись кадра
        video_writer.write(annotated_frame)

    # Освобождение ресурсов
    caption.release()
    video_writer.release()
    cv2.destroyAllWindows()
    return True
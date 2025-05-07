# MTS Dreambooth

## Обзор

Этот проект предназначен для обработки изображений и обучения моделей с использованием техник Dreambooth, LoRA и FLUX. Берём фото, обрезаем по человеку, ресайзим до 1024. Используем один текстовый промпт на все фото. Обучаем LoRA высокого ранга (128 и выше) и batch size 1 с помощью `ai-toolkit`. Оценить результат можно с 2000 шага.

## Скрипты

- `process_images_yolo.py`: Обрабатывает JPG-изображения: удаляет неподходящие (меньше 1024x1024, без людей по YOLOv8), обрезает до квадрата вокруг самого большого человека (+20%), изменяет размер до 1024x1024 и перезаписывает.
- `process_captions.py`: В .txt файлах в `./processed3/` заменяет "Photo of MARK" на "a photo of ohwx man".
- `train_dreambooth_lora_flux.py`: Обучает DreamBooth LoRA для FLUX с помощью Hugging Face diffusers. Позволяет обучить модель новому понятию. Поддерживает чекпоинты, валидацию, логирование.
- `train_dreambooth_lora_flux_advanced.py`: Расширенная версия предыдущего скрипта. Добавляет поддержку "Pivotal Tuning" / Textual Inversion для создания новых токен-эмбеддингов (CLIP, T5) и более детальной карточки модели.
- `make_dataset_for_diffusers.py`: Готовит датасет для Diffusers. Изображения из `processed/` уменьшаются (макс. сторона 1280px), сохраняются в `dataset/train/` как `0.jpg` и т.д. В описаниях "MARK" заменяется на "p3rs0n". Создает `dataset/train/metadata.jsonl`.
- `process_images_make_captions.py`: Обрабатывает изображения из `input/`. Генерирует описания с Qwen2.5-VL-32B-Instruct. Изображения изменяются (макс. сторона 1280px, кратно 28). В описаниях "man", "hair" заменяются на "MARK", добавляется "a photo of". Результаты (JPG, TXT) в `processed/`.
- `download_test_dataset_for_diffusers.py`: Загружает тестовый датасет "LinoyTsaban/3d_icon" из Hugging Face Hub в `./3d_icon`.
- `train_dreambooth_diffusers.sh`: Shell-скрипт для запуска обучения DreamBooth LoRA (`train_dreambooth_lora_flux.py` или `_advanced.py`) через `accelerate launch` с предопределенными параметрами.
- `archive.sh`: Создает ZIP-архив `vastai_test.zip` с каталогами `./dataset`, `./processed` и некоторыми скриптами.

(Добавьте сюда другие важные скрипты или файлы)

## Подмодули

Клонирование с подмодулями:
```bash
git clone --recurse-submodules <URL-репозитория>
```
Инициализация/обновление существующих:
```bash
git submodule update --init --recursive
```

## Настройка и установка

(Инструкции по настройке: версия Python, venv, зависимости)

Пример:
```bash
# Виртуальное окружение (рекомендуется)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# Установка зависимостей
pip install -r requirements.txt
```

## Использование

(Как запускать скрипты или использовать проект. Примеры команд.)

Пример:
```bash
python train_dreambooth_lora_flux.py --ваши --аргументы
```

## Конфигурация

(Описание конфигурационных файлов или переменных окружения.)

## Участие в проекте

(Необязательно: руководство по участию.)

## Лицензия

(Укажите лицензию, например, MIT, Apache 2.0.) 
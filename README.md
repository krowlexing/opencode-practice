# opencode-practice

1. Для распознавания людей использовал нейросеть [yolov8](https://github.com/ultralytics/ultralytics)

2. Для поиска групп людей использовал границы с предыдущего шага, файл [group.py](./group.py)

3. Для распознавания каски пытался в верхней трети человека найти оранжевое(или зеленое) пятно, файл [helmet.py](./helmet.py)

### Установка:

```sh
pip install -r requirements.txt
```

### Запуск:

```sh
python detect.py
```

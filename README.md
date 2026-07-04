# Улучшение изображений · ML в браузере

Система автоматического улучшения изображений (яркость, контрастность, цветность)
с помощью ML-модели, работающей **целиком в браузере пользователя**. Изображения
не покидают устройство.

## Принцип работы

ИИ (лёгкая CNN-регрессия) анализирует уменьшенную копию изображения и подбирает
3 параметра коррекции — **яркость, контраст, насыщенность**. Затем быстрый
WebGL-шейдер применяет эти параметры к изображению в полном разрешении. Вся
тяжёлая работа выполняется в Web Worker, поэтому интерфейс не блокируется.

> Оценка качества и эталонный пул: [docs/EVALUATION.md](docs/EVALUATION.md)

## Запуск

```bash
npm install
npm run dev       # режим разработки (http://localhost:5173)
npm run build     # сборка в dist/
npm run preview   # предпросмотр собранного билда
```

## API

```ts
import { ImageEnhancer } from './src/api/ImageEnhancer';

const enhancer = new ImageEnhancer();

// Событие изменения статуса задачи
enhancer.addEventListener('statuschange', (e) => {
  const { taskId, status, progress } = e.detail;
});

const id = await enhancer.createTask(file);    // постановка задачи
const status = enhancer.getStatus(id);          // статус и прогресс
await enhancer.abortTask(id);                   // прерывание
const blob = await enhancer.getResult(id);      // готовое изображение
```

| Метод | Назначение |
|-------|-----------|
| `createTask(input, options?)` | Постановка задачи, возвращает `taskId` |
| `getStatus(taskId)` | Текущий статус и прогресс |
| `abortTask(taskId)` | Прерывание задачи |
| `getResult(taskId, as?)` | Готовое изображение (`blob` / `bitmap` / `dataurl`) |

Событие `statuschange` несёт `{ taskId, status, progress, params }`.

## Поддерживаемые форматы

JPG · PNG · BMP (нативно) · HEIC (через libheif-wasm, грузится лениво).

## Демо-примеры

На стартовом экране есть кнопки «Попробуйте на примере» (тёмное / дымка / блёклые
цвета) — работу ИИ видно в один клик, без поиска своего файла. Примеры
генерируются скриптом `training/make_demo_samples.py`.

## Технологии

Vite · TypeScript · TensorFlow.js · WebGL · Web Workers · OffscreenCanvas.

## Статус

| Этап | Статус |
|------|--------|
| ТЗ и архитектура | ✅ |
| Каркас + API + пайплайн | ✅ |
| Обучение ML-модели | ✅ |
| Эталонный пул | ✅ |
| Проверка форматов и 15 Мпк | ✅ |
| Оценка качества | ✅ ([отчёт](docs/EVALUATION.md)) |
| Бенчмарк (скорость/память/CPU) | ✅ ([отчёт](docs/BENCHMARK.md)) |
| Деплой на хостинг | ⏳ |

Параметры подбирает обученная ИИ-модель (TensorFlow.js, `public/model/`); при её
отсутствии система прозрачно откатывается на классический авто-алгоритм.
Обучение и оценка — в каталоге [`training/`](training/).

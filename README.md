# Smart Parking

Сервис определяет занятость парковочных мест по изображению.

Проект состоит из 4 частей:
- `api` на FastAPI принимает запросы и отдаёт статусы задач.
- `db` на PostgreSQL хранит парковки, места, задачи анализа и результаты.
- `redis` работает как брокер очереди.
- `worker` на Celery забирает задачу из очереди и запускает анализ изображения в фоне.

## Что реализовано

По требованиям закрыты все 6 пунктов:
- есть `Dockerfile` и `docker-compose.yml`;
- в `docker-compose.yml` поднимаются `api`, `db`, `redis`, `worker`;
- Celery настроен на Redis;
- `POST /analyses` ставит задачу в очередь;
- `POST /parking-lots/{lot_id}/spots/import` импортирует парковочные места из JSON;
- `GET /analyses/{task_id}` возвращает статус и результаты;
- результаты анализа сохраняются в отдельную таблицу `analysis_results`;
- для API и фоновой обработки есть автотесты.

## Как это работает

1. Клиент создаёт парковку и парковочные места.
2. Клиент либо импортирует полигоны из JSON, либо отправляет JSON сразу вместе с изображением в `POST /analyses`.
3. API сохраняет файл, создаёт запись в `analysis_tasks`, отправляет `task_id` в Redis через Celery и сразу отвечает `202 Accepted`.
4. Celery worker получает задачу, запускает детекцию машин и определяет статус каждого места.
5. Результаты сохраняются в таблицу `analysis_results`, а задача получает статус `completed` или `failed`.
6. Клиент опрашивает `GET /analyses/{task_id}` и получает текущий статус и результат.

## Структура данных

Основные таблицы:
- `parking_lots` — парковки;
- `parking_spots` — парковочные места;
- `analysis_tasks` — задачи на анализ изображения;
- `analysis_results` — результаты анализа по каждому месту.

## Требования

Нужно установить:
- Docker Desktop с включённым Linux engine;
- свободные порты `8000`, `5432`, `6379`.

Также в проекте должны существовать модели для детекции, на которые ссылается `parking_engine/src/utils/config.py`.

## Запуск через Docker

Из корня проекта:

```powershell
docker compose up --build
```

После старта будут доступны:
- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

Проверить, что контейнеры живы:

```powershell
docker compose ps
```

Проверить здоровье API:

```powershell
curl.exe http://localhost:8000/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

## Как проверить работу руками

### 1. Создать парковку

```powershell
curl.exe -X POST "http://localhost:8000/parking-lots" `
  -H "Content-Type: application/json" `
  -d "{\"name\":\"Demo parking\",\"coordinates\":[[55.75,37.61],[55.76,37.61],[55.76,37.62],[55.75,37.62]]}"
```

Сохраните `id` парковки из ответа, например `1`.

### 2. Создать парковочные места

```powershell
curl.exe -X POST "http://localhost:8000/spots" `
  -H "Content-Type: application/json" `
  -d "{\"lot_id\":1,\"spot_number\":\"A1\",\"polygon\":[[0,0],[100,0],[100,100],[0,100]],\"status\":\"free\"}"
```

```powershell
curl.exe -X POST "http://localhost:8000/spots" `
  -H "Content-Type: application/json" `
  -d "{\"lot_id\":1,\"spot_number\":\"A2\",\"polygon\":[[110,0],[210,0],[210,100],[110,100]],\"status\":\"free\"}"
```

### 2a. Вместо ручного создания мест импортировать полигоны из JSON

Если у вас уже есть `parking_zone.json` в COCO-подобном формате, можно не создавать `spots` вручную:

```powershell
curl.exe -X POST "http://localhost:8000/parking-lots/1/spots/import" `
  -F "replace_existing=true" `
  -F "polygons_file=@C:\full\path\to\parking_zone.json"
```

Ожидаемый ответ:

```json
{
  "lot_id": 1,
  "imported_spots": 28,
  "image_id": 1
}
```

### 3. Поставить задачу на анализ

```powershell
curl.exe -X POST "http://localhost:8000/analyses" `
  -F "lot_id=1" `
  -F "image=@C:\full\path\to\parking.jpg"
```

Если хотите в одном запросе и загрузить JSON, и сразу запустить анализ:

```powershell
curl.exe -X POST "http://localhost:8000/analyses" `
  -F "lot_id=1" `
  -F "image=@C:\full\path\to\parking.jpg" `
  -F "polygons_file=@C:\full\path\to\parking_zone.json"
```

Ожидаемый ответ:

```json
{
  "task_id": "uuid",
  "status": "queued"
}
```

### 4. Получить результат

Подставьте `task_id` из предыдущего ответа:

```powershell
curl.exe "http://localhost:8000/analyses/<task_id>"
```

Пока задача обрабатывается, статус будет `queued` или `processing`.

После завершения ожидается ответ такого вида:

```json
{
  "task_id": "uuid",
  "status": "completed",
  "lot_id": 1,
  "result": [
    {
      "spot_id": 1,
      "spot_number": "A1",
      "status": "occupied"
    },
    {
      "spot_id": 2,
      "spot_number": "A2",
      "status": "free"
    }
  ],
  "error_message": null,
  "created_at": "2026-04-21T21:00:00",
  "updated_at": "2026-04-21T21:00:03"
}
```

## Как проверить, что очередь реально работает

Откройте логи воркера:

```powershell
docker compose logs -f worker
```

Когда отправите `POST /analyses`, в логах воркера должна появиться обработка задачи Celery.

Логи API:

```powershell
docker compose logs -f api
```

## Автотесты

Локально в текущем окружении тесты проходят:

```powershell
python -m pytest -q
```

Покрыто:
- создание парковки;
- создание места;
- постановка задачи анализа;
- получение результата анализа;
- сохранение результатов в новую таблицу;
- выполнение фоновой задачи Celery.

## Полезные команды

Остановить проект:

```powershell
docker compose down
```

Остановить проект и удалить volumes:

```powershell
docker compose down -v
```

Пересобрать контейнеры:

```powershell
docker compose up --build --force-recreate
```

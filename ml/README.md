# ML-пайплайн прогнозирования стоимости перевозок

## Что делает пайплайн

Пайплайн извлекает очищенные ставки из Core, готовит признаки и обучает
`CatBoostRegressor` для прогноза стоимости рейса в рублях. Модель, метрики и
прогнозы сохраняются в файлы; результаты test и future публикуются в PostgreSQL.

```text
core.prices → признаки → CatBoost → оценка на test → файлы → mart.ml_price_predictions
```

## Запуск через Docker Compose

Все команды выполняются из корня проекта. Первоначальный запуск:

```bash
docker compose up --build
```

Сервис `ml` автоматически выполняется после успешного ETL. Подключение к БД
настроено в Compose, результаты сохраняются в `ml/artifacts` на компьютере.
Дата среза задаётся `MART_AS_OF_DATE`, потоки и seed — `ML_THREAD_COUNT` и
`ML_RANDOM_SEED`; значения по умолчанию приведены в [.env.example](../.env.example).
Общая инструкция — в [корневом README](../README.md#быстрый-запуск).

Повтор всей ML-цепочки при работающем PostgreSQL и заполненном Core:

```bash
docker compose run --rm --no-deps --build ml
```

Просмотр JSON-логов с числами строк, метриками и результатом публикации:

```bash
docker compose logs ml
```

При запуске через `docker compose run` логи видны в терминале запуска.

<details>
<summary>Отдельный запуск этапов</summary>

Для следующих команд PostgreSQL должен работать, Core — быть загружен,
а образ `ml` — собран первоначальным запуском. Замените `<UUID>` в путях
на имя завершённого набора или модели из логов. Используются пути внутри
контейнера `/app/ml/artifacts`, соответствующие `ml/artifacts` на компьютере.

Только подготовка признаков:

```bash
docker compose run --rm --no-deps --build ml-features
```

Обучение по готовому набору без публикации в БД:

```bash
docker compose run --rm --no-deps --entrypoint uv ml run --no-sync python -m ml.train_model --dataset-dir "/app/ml/artifacts/dataset-<UUID>" --output-dir /app/ml/artifacts
```

Публикация готовых прогнозов без переобучения:

```bash
docker compose run --rm --no-deps --entrypoint uv ml run --no-sync python -m ml.load_predictions --model-dir "/app/ml/artifacts/model-<UUID>"
```

Построение графиков по готовой модели; результат — в `ml/artifacts/plots-<UUID>`:

```bash
docker compose run --rm --no-deps --entrypoint uv ml run --no-sync python -m ml.plot_report --model-dir "/app/ml/artifacts/model-<UUID>" --output-dir "/app/ml/artifacts/plots-<UUID>"
```

</details>

## Данные и признаки

**Зерно — одна исходная ставка `core.prices.source_price_id`.** Данные извлекаются
из Core и справочников Source; `price_dates` и агрегаты Mart в ML-набор не входят.
История отбирается по `is_fact`, типам `1,2,3,4,5,9`, простому маршруту,
положительным цене и расстоянию, `outlier_mark=false`.

Для стандартного среза `2026-10-26`:

| Выборка | Период по дате погрузки | Назначение |
| --- | --- | --- |
| Train | 26.07–04.10.2026 включительно | Обучение и внутренняя validation |
| Test | 05.10–26.10.2026 включительно | Независимая оценка |
| Future | После 26.10.2026 | Прогноз без известного факта |

Для другого среза начало train — минус 3 календарных месяца, начало test —
минус 21 день. Train и test должны быть непустыми. Future сохраняет фильтры
факта, типа, маршрута и расстояния, но на этапе ML не фильтруется по цене или
`outlier_mark`. Будущая цена скрывается в SQL и повторно в Python.

Используются **28 признаков**:

| Группа | Число | Состав |
| --- | ---: | --- |
| Физические | 7 | Расстояние, вес, объём, координаты отправления и назначения |
| Категориальные | 5 | Два региона, основной кузов, весовая и объёмная категории |
| Допустимые кузова | 6 | Бинарные флаги совместимых групп |
| Календарные | 10 | Год, месяц, день недели, день года и циклические sin/cos |

Основной кузов берётся из ставки, допустимые группы — из `price_body_types`;
при отсутствии связей используется основной тип. Несколько кузовов дают флаги
в одной строке. Города дедуплицируются по минимальному `cities.id` на `gc_id`;
регион извлекается из полного названия, неизвестный — `UNKNOWN`.
Отсутствующий город останавливает подготовку, некорректные координаты становятся `NaN`.

Состав признаков задан явно: цены и их производные, `outlier_mark`, групповые
ценовые агрегаты, идентификаторы и названия городов в модель не передаются.
Правила — в [SQL извлечения](sql/extract_dataset.sql) и [features.py](features.py).

## Обучение и оценка

1. Проверить входной набор, контрольные суммы файлов, признаки и границы дат.
2. Выделить последние 14 дней train под validation; для стандартного среза это
   21.09–04.10.2026. По validation RMSE выбрать число деревьев с early stopping.
3. Заново обучить итоговую модель на всём train с выбранным числом деревьев.
4. Рассчитать на test **RMSE и MAE в рублях, MAPE в процентах и R²**; сравнить
   с baseline — постоянным прогнозом средней цены train.
5. Той же итоговой моделью получить прогнозы future, сохранить CBM и проверить
   совпадение прогнозов после обратной загрузки модели.

Целевая переменная — исходная `price_rub` без логарифмирования. Test не участвует
в выборе числа деревьев и обучении. Категории передаются в CatBoost строками.
Параметры заданы в [training.py](training.py), последовательность — в
[train_model.py](train_model.py). Численные результаты и графики приведены в
[корневом README](../README.md#ml-и-результаты).

## Результаты и загрузка в PostgreSQL

Каждый запуск создаёт отдельные каталоги в `ml/artifacts`:

| Результат | Основные файлы |
| --- | --- |
| `dataset-<UUID>/` | `manifest.json`; `train`, `test`, `future` с `X.parquet`, `y.parquet`, `metadata.parquet` |
| `model-<UUID>/` | `model.cbm`, `metrics.json`, `selection_history.csv`, `feature_importance.csv` |
| Прогнозы в каталоге модели | `predictions_test` и `predictions_future` в Parquet и CSV |
| Отчёт запуска | `training_report.json`: параметры, версии, происхождение и контрольные суммы |

`manifest.json` и `training_report.json` записываются последними и обозначают
завершённые артефакты. Незавершённые или повреждённые наборы отклоняются.

Таблица **`mart.ml_price_predictions`** содержит ID ставки, дату, маршрут, кузов,
категории, фактическую и прогнозную цену, ошибки и происхождение результата.
Ключ — `(model_version, as_of_date, source_price_id)`. Для future факт и обе ошибки
равны `NULL`; `abs_error_rub` и `ape_pct` вычисляет PostgreSQL.

Загрузчик проверяет полный артефакт test/future и выполняет **COPY → DELETE → UPSERT
одной транзакцией**. Удаляются исчезнувшие ID только выбранной модели и среза;
остальные результаты сохраняются. При ошибке вся операция откатывается.
Повтор того же артефакта не меняет строки, включая `loaded_at`; более старый
артефакт не перезаписывает свежий. Новый запуск обучения может обновить
происхождение прогноза, сохранив единственность ключа.

DDL применяется автоматически при первой инициализации PostgreSQL.
Подробности — в [DDL](../sql/06_mart_ml_price_predictions.sql) и
[загрузчике](load_predictions.py); для существующей БД — в
[инструкции Docker](../docker/README.md).

## Проверка результата

Запуск всего набора тестов:

```bash
docker compose run --rm --build tests
```

Runner сам создаёт и удаляет отдельные временные БД. ML-тесты проверяют зерно,
фильтры, несколько кузовов, отсутствие утечки будущей цены, временное разбиение,
метрики, загрузку модели и независимость обучения от test-цен. Тесты публикации
проверяют повтор, обновление/удаление, изоляцию моделей/срезов и откат при ошибке.

SQL для Query Tool pgAdmin: количество прогнозов по модели, срезу и выборке:

```sql
SELECT model_version, as_of_date, dataset_split, count(*) AS rows
FROM mart.ml_price_predictions
GROUP BY model_version, as_of_date, dataset_split
ORDER BY as_of_date DESC, model_version, dataset_split;
```

Примеры прогнозов последнего загруженного среза:

```sql
WITH latest AS (
    SELECT model_version, as_of_date
    FROM mart.ml_price_predictions
    ORDER BY loaded_at DESC, model_version, as_of_date DESC
    LIMIT 1
)
SELECT p.source_price_id, p.loading_date, p.dataset_split,
       p.loading_city_name, p.unloading_city_name,
       bg.name_ru AS body_group, p.actual_price_rub,
       p.predicted_price_rub, p.abs_error_rub, p.ape_pct
FROM mart.ml_price_predictions p
JOIN latest USING (model_version, as_of_date)
JOIN mart.body_groups bg USING (body_group)
ORDER BY p.loading_date, p.source_price_id
LIMIT 20;
```

При сравнении метрик выбирайте одну модель и дату среза. Дополнительные запросы —
в [корневом README](../README.md#sql-для-проверки-витрин).

## Ограничения

- Данные синтетические; метрики test не гарантируют качество на реальных перевозках.
- Короткий период одного года не позволяет проверить годовую сезонность и межгодовой тренд.
- Core и справочники хранят текущее состояние: `as_of` ограничивает бизнес-даты,
  но не восстанавливает прежние версии. Дата завершения рейса отсутствует;
  `is_fact` и дата погрузки служат приближением доступности факта.
- Future приходит из Core и уже прошёл его ценовые/валютные проверки. Для заявок
  совсем без цены нужен отдельный входной контракт. Без факта качество future не измеряется.

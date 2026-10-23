# Dota2 Analyzer

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![ML: scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)

Аналитический инструмент для **Dota 2**: собирает реальные матчи про-сцены с открытого
Cito API, считает метрики, строит ML-модель для предсказания победителя и показывает
результат в веб-интерфейсе Streamlit.

Построен по архитектуре «CS2 Analyzer» (тот же пайплайн: scraper → features → model → UI),
но адаптирован под Dota 2.

## Возможности

- **Сбор данных** — реальные завершённые матчи Dota 2 с **Cito API** (без рисков скрапинга),
  плюс исторические матчи The International из Liquipedia.
- **Фичи** — recency-weighted форма команд, ELO-рейтинги, strength of schedule,
  официальный мировой рейтинг команд (Cito), свежесть состава и др.
- **ML** — ансамбль (RandomForest / GradientBoosting / LogisticRegression),
  валидация через `TimeSeriesSplit`, симметричность прогноза к порядку команд.
- **Streamlit UI** — страницы: Прогноз матча, Форма команд, Backtest.
- **Автообновление** — раз в час докачивает новые матчи и переобучает модель.

## Быстрый старт

### Двойной клик

Дважды кликните **`Запуск_Dota2_Анализатора.bat`** в корне проекта. Он сам активирует
окружение, запустит Streamlit и откроет браузер на `http://localhost:8501`.

### Вручную

```bash
.venv\Scripts\activate
streamlit run app.py
```

### Первый запуск с нуля (новый ПК)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python fetch_dota_cito.py     # собрать матчи + рейтинги с Cito
python -m src.features
python -m src.model
streamlit run app.py
```

## Конфигурация

Ключ Cito API (бесплатный план 500 запросов/мес) вставляется в `.env`:

```
CITO_API_KEY=cito_...
```

Без ключа сбор с Cito не работает, но проект может запускаться на синтетике
(`CS2_SCRAPER_MODE=mock`).

## Данные

- `data/raw/matches.csv` — реальные матчи (~290, обновляются авто-фетчем).
- `data/processed/cito_rankings.parquet` — мировой рейтинг команд Dota.
- `data/processed/features.parquet` — фичи.
- `models/model.pkl` — обученный ансамбль + метаданные.

## Тесты

```bash
python -m pytest tests -v
```

## Честно о точности

Dota 2 — высокостохастичная игра (многое решает пик-фаза, форма в день матча).
Модель даёт AUC ~0.72 / accuracy ~0.64 на честной temporal-валидации — на уровне
лучших открытых моделей, но проценты стоит воспринимать как оценку, а не гарантию.
Прогноз не зависит от порядка команд (симметричен).
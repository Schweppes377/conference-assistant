# Руководство пользователя offline-прототипа

## 1. Окружение

Windows/PowerShell, Python 3.12, существующая `.venv`. Работайте из корня:

```powershell
cd C:\Users\maree\PycharmProjects\conference-assistant
.\.venv\Scripts\Activate.ps1
```

Для команд ниже активация необязательна: `run-direct.ps1` сам выбирает `.venv`.
Launcher изолирует proxy только у запуска Python и восстанавливает окружение
PowerShell; Happ, браузер и Codex не отключаются. Не переустанавливайте рабочие
зависимости/модели. Для локальных тестов: `python -m pip install -r requirements-dev.txt`
через тот же launcher, если pytest ещё отсутствует.

В локальном `.env` нужны `HF_TOKEN` (pyannote) и `SBER_ULTRA` (Authorization Key
GigaChat). Ключи не показывать и не коммитить. LLM: GigaChat-3-Ultra,
GIGACHAT_API_PERS; ASR по умолчанию medium/ru, embedding multilingual-e5-small.
Проверка сертификата отключена только у GigaChat-клиента; не меняйте глобальный SSL.

## 2. Нормативные PDF

Положите законно полученные, читаемые PDF в `data/documents/`.
В `sources.json` сохраните file/title/designation/source_url/status/edition_year/
has_extractable_text. Проверяйте редакцию и актуальность самостоятельно.
Для сканов без текста OCR не реализован; HTML нельзя переименовывать в PDF.

## 3. Индексация

```powershell
.\run-direct.ps1 -m rag.index
.\run-direct.ps1 -m rag.search "требования к эвакуационным выходам"
```

Рабочий индекс уже существует; не индексируйте его при каждой демонстрации.
Индексация заменяет chunks конкретного файла, не создавая дублей. Она запускает
embeddings, но не LLM. Chunks имеют документ, страницу PDF и индекс фрагмента.

## 4. Аудио

Сохраните файл, например `data/audio/construction_meeting.wav`. Система обрабатывает
готовый файл, не микрофон и не поток Zoom/Teams. Декодирование поддерживает форматы,
читаемые faster-whisper/PyAV; ошибочный или пустой файл диагностируется отдельно.

## 5. Анализ

```powershell
.\run-direct.ps1 -m app.analyze_meeting data\audio\construction_meeting.wav --debug
```

Шесть стадий: распознавание → говорящие → стенограмма → вопросы → RAG/LLM → отчёт.
Без `--debug` вывод краткий; debug показывает timings, calls/tokens и кандидатов.
Полный запуск расходует GigaChat. Если нужна только стенограмма:

```powershell
.\run-direct.ps1 main.py data\audio\construction_meeting.wav
```

## 6. Сообщения

`partial_information` — контекст отвечает лишь частично; `requirement_found` —
требование найдено; `potential_noncompliance` — потенциальное несоответствие,
не юридическое заключение; `insufficient_information` — контекста недостаточно;
`processing_error` — локальная проверка/обработка не пройдена; `pending` — ожидает
продолжения. CPU fallback возможен, но заметно медленнее. Rate limit/timeout
останавливает дальнейшие LLM-вызовы и сохраняет progress.

## 7. Результаты

`data/results/*_transcript.json`, `*_analysis.json`, `*_report.md`, `*_progress.json`.
CLI печатает точные пути. JSON UTF-8 содержит структурированные данные; Markdown
содержит только использованные источники. Новые analysis/progress сохраняют raw
diarization и длительность аудио для оценки. Старые файлы не переписываются.

## 8. Продолжение

```powershell
.\run-direct.ps1 -m app.analyze_meeting --resume "data\results\ИМЯ_progress.json" --debug
```

Скопируйте точный путь из CLI. Готовые ASR/diarization, detection, retrieval и ответы
не повторяются. Provider/model и настройки должны соответствовать checkpoint.
Для несовместимой модели начните новый эксперимент без повторного аудио:

```powershell
.\run-direct.ps1 -m app.analyze_meeting --transcript "data\results\ИМЯ_transcript.json" --debug
.\run-direct.ps1 -m app.analyze_meeting --issues "data\results\ИМЯ_analysis.json" --reanswer --debug
```

Второй вариант повторяет ответы, но не detection. Не используйте `--reanswer` ради
обычного resume. Бюджет задаётся `--max-llm-calls`/MAX_LLM_CALLS_PER_RUN; максимум
два generation attempts на окно или вопрос, без вложенных retry.

## 9. Проверка отчёта и локальная оценка

В отчёте: сводка → время обсуждения → исходный фрагмент → normalized question →
ответ → статус → документ/страница/chunk. Проверяйте применимость нормы по PDF.
Доступность числа в источнике сама по себе не доказывает правильность всего ответа.

```powershell
.\run-direct.ps1 -m pytest -q
.\run-direct.ps1 -m evaluation.run
```

Эти команды не вызывают GigaChat. Для WER/DER/F1/Hit@K сначала разметьте независимый
эталон по `evaluation/README.md`. `SKIPPED` означает отсутствие эталона, не провал.
CSV/JSON из `evaluation/results/` предназначены для таблиц ВКР.

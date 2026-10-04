# Локальные измерения для ВКР

Это отдельный контур, не production inference. Команда `python -m evaluation.run`
не запускает ASR, diarization, detection, LLM или embeddings по умолчанию.
Она оценивает **сохранённую гипотезу**, а не получает новый ответ.

## Разметка

Скопируйте `datasets/construction_meeting.template.json` под новым именем.
Шаблон ссылается на настоящий baseline, но все references оставлены `null`.
`datasets/dataset.schema.json` описывает формат. Пути в JSON — от корня проекта.
`null` означает «не размечено», `[]` — «размечено, событий нет»; это разные случаи.

1. Прослушайте запись независимо от машинного результата. `reference_transcript`
   — дословный русский текст строкой; либо массив `start/end/speaker/text`.
   Не копируйте машинную стенограмму как эталон. Сохраните числа, термины и ошибки
   произношения; заранее зафиксируйте политику для слов-паразитов. ASR reference
   должен покрывать весь файл: partial/UEM scoring реализован только для DER.
2. `reference_diarization` — все речевые интервалы `start/end/speaker` в секундах
   от начала файла. Один человек всегда имеет одну метку; имена произвольны.
   Тишину не размечайте как речь. Пересечения оформляйте отдельными интервалами
   двух говорящих. `evaluation_regions` задаёт реально полностью размеченные
   интервалы; для частичной разметки не оценивайте остальную запись.
3. `reference_issues` — независимый список нормативных вопросов с собственными
   `id`, `question`, `start/end` обсуждения. Для `matching.mode=segments` вместо
   времени укажите `segment_ids`, относящиеся к **той же сохранённой стенограмме**.
   После нового ASR такая привязка может стать неверной; временная предпочтительнее.
4. Для поиска у вопроса добавьте `expected_sources`: точное `designation` или
   `document`, при наличии `page` (номер PDF, начиная с 1) и `chunk_id`.
   Идентификатор evidence имеет вид `filename.pdf:pPAGE:cINDEX`, а не внутренний
   SHA-256 Chroma. Возьмите его из debug/progress и проверьте содержание по PDF.
   Несколько источников трактуются как допустимые альтернативы для Hit@K.
5. `expected_status` и `expected_answer_facts` заполняет человек после проверки
   норматива. Status можно сравнить автоматически для сопоставленных вопросов;
   смысловую корректность facts оценивает специалист, не новая LLM.

Не ставьте reference по текущим восьми статусам: это результаты baseline,
а не доказательство истинности требований. Полезна независимая разметка двумя
людьми и разбор расхождений. Все экспериментальные thresholds фиксируйте заранее.

## Метрики и ограничения

- WER = (S+D+I)/N; CER аналогично. Lowercase, пунктуация заменяется пробелами,
  лишние пробелы удаляются. `ё` не заменяется на `е`, числа не переводятся в слова,
  единицы не исправляются. CER считается без пробелов. Выравнивание Levenshtein;
  при равной стоимости порядок substitution/deletion/insertion фиксирован.
- DER: `pyannote.metrics.DiarizationErrorRate`, Hungarian permutation mapping,
  collar 0.25 с, overlap учитывается. Сохраняются настройки и components. Нужны
  **исходные** интервалы diarization; интервалы Whisper с назначенной меткой не
  эквивалентны им. Старый baseline не содержит raw diarization — DER пропускается.
- Detection: максимальное 1:1 matching по temporal IoU >= 0.5; альтернативно
  Jaccard segment IDs >= 0.5. Считаются TP/FP/FN/P/R/F1. Это оценка локализации,
  не понимания смысла вопроса: два разных вопроса в одном интервале могут совпасть.
  Формулировки и применимость требований отдельно оцениваются вручную.
- RAG: Hit@1/3/5 и MRR. Предпочтительно порядок `prepared.context_sources` из
  progress — фактически переданный контекст. Если есть только analysis,
  используются сохранённые reranked candidates и это явно отмечается; результаты
  этих двух режимов не следует смешивать. Неразмеченные/недоступные запросы не
  входят в знаменатель, coverage отражается числом evaluated/provided queries.
  MRR относится к переданному ограниченному ranking, а не всему корпусу;
  глубина ranking сохраняется для каждого запроса.
- Latency берётся из сохранённых timings, длительность — из нового поля результата
  или заголовка WAV (без декодирования). ASR RTF = asr_sec / audio_duration_sec;
  full RTF считается только для полного запуска. Для resume `total_sec` старой
  архитектуры не является суммарным wall time всех запусков — full RTF пропускается.
  Нулевые/непредоставленные знаменатели дают `null`, не фиктивные нули.
  Нулевые placeholders timings для невыполненных этапов экспортируются как null,
  а не как измерения мгновенного inference.
- Reliability: локальные pytest pass/fail/skipped и recovery scenarios, не SLA.
  Synthetic fixtures проверяют evaluator, не качество реального совещания.

## Команды (из корня проекта, PowerShell)

```powershell
.\run-direct.ps1 -m pytest -q
.\run-direct.ps1 -m pytest -q -m "not integration"
.\run-direct.ps1 -m pytest -q -m integration
.\run-direct.ps1 -m evaluation.run
.\run-direct.ps1 -m evaluation.run --dataset evaluation\datasets\my_control.json --analysis data\results\MY_analysis.json --progress data\results\MY_progress.json --output-dir evaluation\results\my_control
```

По умолчанию summary — `evaluation/results/baseline/summary.json`, таблицы — CSV
UTF-8 BOM рядом; pytest сохраняет `evaluation/results/reliability.json`.
Для нового эксперимента используйте другой `--output-dir`, чтобы сохранить предыдущий.

Опционально, **только по явному запросу**, `--live-retrieval` оценивает reference
questions через существующий `QuestionAnsweringService.prepare()`/RAG и текущую
конфигурацию. LLM не создаётся; downloads запрещены HF offline flags. Нужны готовый
индекс и локальный embedding cache; режим не запускался при подготовке baseline.
Время evaluation не является временем нового полного анализа.

Manual/e2e — существующая команда приложения, запускается пользователем отдельно:

```powershell
.\run-direct.ps1 -m app.analyze_meeting data\audio\construction_meeting.wav --debug
```

Она расходует токены GigaChat; ни pytest, ни обычная evaluation её не вызывают.

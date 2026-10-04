# Техническая шпаргалка для главы 3

Это карта реального кода, не готовый текст ВКР. Реализован **offline**-прототип:
готовый аудиофайл → ASR → diarization → timestamp merge → detection → существующий
RAG → GigaChat → локальная validation → сохранение состояния и отчёт.
Никакого захвата микрофона/системного звука, Zoom/Teams, streaming или realtime UI нет.

## Рабочий baseline и границы доказательств

Проверенный сохранённый результат:
`data/results/construction_meeting_20261004_233227_076424_analysis.json`;
checkpoint: `construction_meeting_20261004_233031_181110_progress.json`.
Это ранее выполненный пользователем запуск, не новый интеграционный эксперимент.
Статусы не изменялись при подготовке инструментов оценки.

- 56 реплик, SPEAKER_00/SPEAKER_01, 8 вопросов, processing_error=0.
- 9 successful LLM calls, corrective_retries=0, 27330 total tokens.
- Статусы: ISSUE_001 partial_information; ISSUE_002/003 potential_noncompliance;
  ISSUE_004 partial_information; ISSUE_005/006 requirement_found;
  ISSUE_007 partial_information; ISSUE_008 requirement_found.
- WAV header: 247.616 с. Сохранённый total: 115.81566180009395 с.
  Это wall time именно данного полного запуска; не SLA и не статистика многих запусков.
- WER/DER/F1/Hit@K неизвестны до независимой разметки. «8 вопросов» не означает
  Recall=100%, «0 processing_error» не означает фактическую правильность восьми ответов.

## Соответствие плану

Для каждого подраздела ниже перечислены код/функции, вход/выход, библиотеки,
подходящий скриншот и допустимые результаты. Основной pipeline был реализован
до этой задачи; evaluation и наблюдаемость добавлены отдельно.

| Раздел | Код и основные сущности | Вход → выход | Библиотеки | Скриншот / результаты / готовность |
|---|---|---|---|---|
| 3.1.1 Структура и настройка | `run-direct.ps1`, `sitecustomize.py`, `requirements.txt`, `requirements-dev.txt`, `rag/config.py:RAGConfig`, `llm/config.py:QAConfig` | process env, локальные пути → параметры запуска | Python 3.12, dotenv, truststore | дерево каталогов и команды без `.env`; готово, model defaults не менялись |
| 3.1.2 Распознавание | `asr/transcriber.py:ASRSegment, WhisperTranscriber, _load_model`, `utils/audio.py:load_audio` | WAV/MP3 → mono float32 16 kHz → start/end/text | faster-whisper, CTranslate2, PyAV, torch/NumPy | пример transcript и ASR timing; medium/ru, CUDA float16 или CPU int8, ленивый cache; реализовано |
| 3.1.3 Метки говорящих | `diarization/diarizer.py:PyannoteDiarizer`, `app/transcript.py:merge_transcript` | PCM + HF_TOKEN → turns → текст со speaker | pyannote.audio Community-1, torch | реплики двух спикеров, raw turns; exclusive output с fallback, максимум суммарного overlap; реализовано |
| 3.1.4 Документы и поиск | `rag/pdf_loader.py`, `chunking.py:chunk_pages`, `embeddings.py:SentenceTransformerEmbedder`, `store.py:VectorStore`, `retriever.py:SemanticRetriever`, `reranker.py` | PDF/catalog → pages/chunks → vectors → ranked evidence | PyMuPDF, multilingual-e5-small, ChromaDB | sources.json, page metadata и search debug; 1200/200 chars, E5 query/passage prefixes, document replacement; реализовано |
| 3.1.5 Вопросы и LLM | `app/issue_detection.py:NormativeIssueDetector`, `llm/service.py:QuestionAnsweringService.prepare/answer`, `gigachat_client.py`, `prompts.py` | transcript windows → issues/search_terms → context → citations/status | официальный gigachat 0.2.3, dataclasses, JSON Schema | normalized questions и cited sources; GigaChat-3-Ultra, candidate_k=20/top_k=5, context 12000 chars, temperature .1, validation; baseline неизменён |
| 3.1.6 Состояние/отчёт | `app/progress.py:save_checkpoint/read_input/check_llm_identity`, `meeting_analysis.py`, `reporting.py:save_meeting_analysis`, `llm/runtime.py:RunLLMClient` | промежуточные результаты → atomic progress, analysis, transcript, report | stdlib json/pathlib/dataclasses | четыре файла, pending и usage; завершённые этапы не повторяются, model guard; реализовано |
| 3.2.1 Контрольные данные | `evaluation/datasets/*.json`, `evaluation/data.py:validate_dataset` | независимая ручная разметка → машиночитаемый эталон | stdlib; JSON Schema как спецификация | заполненный датасет (пока шаблон с null); эталон НЕ создан из baseline |
| 3.2.2 Модули/взаимодействие | `conftest.py`, `evaluation/test_*.py`, существующие `app/test_*.py`, `llm/test_*.py`, `rag/test_*.py` | synthetic/mocks → pytest reports | pytest, unittest.mock, локальная Chroma в temp | вывод pytest и reliability.json; сеть/загрузка моделей запрещены; реализовано |
| 3.2.3 Текущая версия | существующие `app/analyze_meeting.py`, новые failure tests; `evaluation/run.py:evaluate` | сохранённый baseline + независимый эталон → summary | stdlib, прежние pipeline libs | реальные статусы восьми вопросов и error cases; manual/e2e отдельно, не запускается pytest |
| 3.2.4 Задержка/восстановление | `PipelineTimings`, `evaluation/data.py:performance_metrics`, `app/test_api_budget.py` | измеренные timings/WAV header, checkpoint → latency/RTF и mock resume assertions | perf_counter, wave, pytest | performance.csv и тест recovery; новая аудиообработка не запускалась; repeat-run distribution пока не измерена |
| 3.3.1 ASR quality | `evaluation/metrics.py:normalize_text/edit_counts/asr_metrics` | reference text + hypothesis transcript → WER/CER/S/D/I/counts | stdlib | asr.csv; evaluator готов, WER реальной записи SKIPPED до разметки |
| 3.3.2 Speaker quality | `evaluation/metrics.py:diarization_metrics`, raw turns в новых analysis | reference turns + raw hypothesis + UEM → DER/components | pyannote.core, pyannote.metrics, Hungarian mapper | diarization.csv; synthetic permutation тест, реальный DER неизвестен; нужен эталон и raw hypothesis |
| 3.3.3 Вопросы/поиск | `metrics.py:match_issues/detection_metrics/retrieval_metrics`, `run.py:evaluate` | reference issues/relevance + сохранённые predictions/context → P/R/F1, Hit@1/3/5/MRR | stdlib; optional существующий RAG | question_detection.csv/retrieval.csv; отсутствующий ground truth → null; matching оценивает локализацию, не смысл |
| 3.3.4 Надёжность | `conftest.py` pytest hooks, `app/test_api_budget.py`, `llm/test_gigachat_client.py`, новые failure tests | предусмотренные ошибки/mocks → passed/failed/skipped/success rate | pytest/unittest.mock | reliability.json/CSV; 429/timeout/save/resume/invalid JSON/validation проверяются без оплаты; не industrial availability |
| 3.4.1 Подготовка/индексация | `docs/USER_GUIDE.md` §§1–3; `rag/index.py:build_index` | PDF/catalog → существующий индекс | PyMuPDF, sentence-transformers, ChromaDB | реальная команда rag.index; повторная индексация не выполнялась в этой задаче |
| 3.4.2 Запуск | `USER_GUIDE.md` §§4–6; `app/analyze_meeting.py:main` | готовый аудиофайл → шесть стадий и сводка | argparse, существующие pipeline libs | обычный CLI; полный анализ запускает пользователь явно, токены расходуются |
| 3.4.3 Продолжение | `USER_GUIDE.md` §8; progress/runtime/meeting_analysis | совместимый progress → только pending этапы | stdlib, общий LLM слой | --resume, --transcript, --issues --reanswer; mismatch не смешивает модели |
| 3.4.4 Проверка отчёта | `USER_GUIDE.md` §§7,9; reporting/evaluation.run | report/analysis/sources → ручная проверка и таблицы | Markdown, JSON, CSV UTF-8 BOM | report с used sources, summary/CSV; экспертная проверка применимости обязательна |

## Проверки и воспроизводимость

`python -m pytest` собирает существующие unittest-тесты, не дублируя numeric/citation/
language/grounding/429/resume tests. Дополнительные pytest-тесты проверяют evaluator,
timestamp merge, структуру metadata, malformed input и наблюдаемость стадий.
Сеть заблокирована, функции загрузки ASR/diarization/embedding-моделей запрещены.
MockTransport официального SDK проверяет формат HTTP, не соединяется с сервисом.
`rag/test_store.py` использует временный индекс с заданными вручную векторами,
не меняет `data/chroma`. Corpus test лишь читает существующие PDF.

Pytest faulthandler plugin отключён только в pytest.ini: на Windows он выводил
native SEH-дампы при обработанной ошибке импорта опционального torchcodec.
Первый прогон несмотря на дампы завершился 108 passed, exit=0. Исключения Python,
failed assertions и exit codes не подавляются; зависимости audio не менялись.

Reliability считается по фактически выполненным тестам. Сохраняются node IDs,
outcome/phase/duration, passed/failed/skipped/collection_errors и exit status.
Неуспешные тесты не маскируются; success rate исключает skipped.

Небольшая необходимая поправка: неверные типы JSON/progress/analysis раньше могли
попасть в AttributeError/TypeError. Добавлены shape guards на входе с ValueError.
Пустой текстовый transcript отклоняется до LLM. Алгоритм checkpoint/resume,
LLM retries, нормативная validation и статусы не изменялись.

## Что использовать для таблиц

| Таблица ВКР | Источник | Что ещё нужно |
|---|---|---|
| Распознавание речи | `evaluation/results/<run>/asr.csv`, summary.asr | независимая дословная reference transcript |
| Диаризация | `diarization.csv`, summary.diarization.details | reference turns + исходные predicted turns; старый baseline их не сохранял |
| Выделение вопросов | `question_detection.csv`, summary.question_detection.matches | reference issues и заранее выбранный anchor/threshold |
| Поиск нормативов | `retrieval.csv`, `retrieval_queries.csv` | expected_sources, желательно точные страницы/chunks |
| Время этапов | `performance.csv`, observed_statistics.timings | доступны реальные baseline timings; новые повторы измерять отдельно |
| Устойчивость | `evaluation/results/reliability.json`, `reliability.csv` | уже фактические pytest outcomes; не трактовать как SLA |
| Статусы baseline | `issues.csv`, исходный analysis.json | это наблюдения, не accuracy/ground truth |

Все input artifact paths и SHA-256 сохраняются в summary; outputs нового
эксперимента направляйте в отдельный `--output-dir`. Численные WER/DER/F1/Hit@K
не добавляйте в ВКР, пока эти reference поля равны null. `expected_answer_facts`
поддерживается форматом, но автоматический semantic fact judge не реализован.

## Что ещё не выполнено

Не размечен реальный эталон, не измерены quality metrics на независимом наборе,
нет серии повторных latency experiments/доверительных интервалов и экспертной
оценки содержательной корректности. Это экспериментальная работа пользователя.
Raw diarization для старого baseline восстановить из transcript нельзя; для DER
нужен будущий явно запущенный audio run с новой сохранённой raw annotation.
Online-функции намеренно вне границ этой версии ВКР.

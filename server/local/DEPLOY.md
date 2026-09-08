# Локальное развертывание

Команды выполняются на `ext` под пользователем `service`.

```bash
cd /home/service/llama
git pull --ff-only origin v2.4-local-editor
cd server/local
bash install_local_stack.sh
cp -n .env.example .env
```

В `.env` выберите модель:

```env
LLM_PRESET=A
LANGUAGETOOL_ENABLED=true
LANGUAGETOOL_URL=http://127.0.0.1:8081
LANGUAGETOOL_ENABLED_CATEGORIES=GRAMMAR,PUNCTUATION,TYPOGRAPHY,TYPOS
NUM_THREADS=16
```

LanguageTool должен работать локально на `8081`. Публичный API запрещен.

```bash
sudo systemctl restart ai-suggester.service
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
journalctl -u ai-suggester.service -n 100 --no-pager
```

`F` оставлен только для экспериментов. Для production используйте `A`; `G` нужен
для проверки детерминированных кандидатов. Текущие незакоммиченные файлы клиента
не копируйте и не добавляйте в коммит сервера.

# Contribuindo

Abra uma issue com o caso que quer resolver e, quando houver bug numérico, a seed e as premissas.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q
```

Os testes não acessam Yahoo. A fixture offline é sintética e deve continuar identificada assim.
Se mexer na calibração, preserve o ajuste de drift e acrescente um exemplo verificável.
Se mexer no visual, gere os dois layouts e confira PNG, MP4 e GIF; mantenha cada GIF abaixo de 10 MB.

Sugestões pequenas: melhorar formatação monetária em valores abaixo de $1K;
acrescentar um exemplo de comparação de volatilidades no notebook;
documentar instalação do ffmpeg no Windows. Discuta antes de adicionar dependências.

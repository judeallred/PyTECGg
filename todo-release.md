Prima di effettuare una PR da `develop` a `main`, assicurati di aver:
- aggiornato il `README`
- lanciato e superato i test con `pytest`
- fatto bump della versione: `.\scripts\bump-version.ps1 0.4.0`
    - ti devi ritrovare modifiche in 2 punti di `pyproject.toml`, 1 punto di `Cargo.toml` e 1 punto di `Cargo.lock`

Una volta completata la PR (e superati i test automatici) le modifiche sono su `main` e puoi procedere a taggare: spostati su `main` e lancia `git tag v0.4.2`, seguito da `git push origin v0.4.2`

Una volta pushato il tag, viene lanciata la GH action che automaticamente builda per tutte le piattaforme (windows, linux, macos) e pusha su PyPI

Alla fine, ricordati di creare la release su GitHub
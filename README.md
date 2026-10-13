# Scanner RSI + rompimento e reteste — Render gratuito

Entrega pronta para substituir a versão anterior num único Web Service Render.

- **RSI:** três ou mais pivots em todos os timeframes configurados; divergências de dois pivots apenas em 4h e 1d.
- **Rompimento:** apenas 4h/1d. Identifica resistências testadas múltiplas vezes com evidência de volume elevado, rompimento acima da zona, sustentação (3 velas posteriores em 4h; 2 em 1d) e posterior reteste. Duas fases com alertas Telegram independentes: **Possível setup** e **Entrada**.
- **Infraestrutura:** um único Web Service, Streamlit e processo de scan, sem base de dados.

## Render

Build: `pip install -r requirements.txt`

Start: `python start.py`

Consulta [LEIA-ME-Render.txt](LEIA-ME-Render.txt) para a configuração detalhada, exemplos de variáveis de ambiente e limitações.

## Testes

```bash
python -m unittest -v test_setups.py
python rsi_divergence_bot.py --once
```

A primeira linha é 100% offline e não envia mensagens. A segunda analisa a exchange e pode enviar mensagens se tiver credenciais configuradas.

> O perfil de volume é uma estimativa baseada em OHLCV, não uma medição direta do volume transacionado em cada preço. Um sinal técnico não equivale a uma ordem de compra.

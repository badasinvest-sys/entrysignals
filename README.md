# V8 — Correção de alertas duplicados INJ/USDT no Render Free

Esta versão substitui **todos** os ficheiros da V7. Mantém os filtros RSI seletivos, confirmação opcional de Open Interest, gráficos leves de breakout/entrada e histórico Streamlit.

## Porque recebias 3 notificações?

- A V7 considerava setups equivalentes somente quando os breakouts distavam **até uma vela**. A mesma resistência podia ser detetada em vários candles próximos e gerar novas entradas.
- O detector conseguia devolver a fase "Possível setup" **depois de já existir uma "Entrada"** para a mesma zona.
- Os ficheiros JSON do Render Free são **efémeros** e podem desaparecer durante deploy/restart. Sem histórico, o programa não sabe o que já enviou.

## Correções V8

1. Identidade da **campanha** de breakout: mesmo mercado + ativo + timeframe, nível de resistência semelhante (tolerância **1,8%** por defeito) e até **16 velas em 4h** ou **8 velas em 1d** entre rompimentos. Esses critérios são intencionalmente conservadores: podem juntar dois breakouts legítimos muito próximos da mesma zona.
2. **No máximo uma entrada** e um "Possível setup" por campanha. Se ambas as fases surgirem pela primeira vez no mesmo scan, envia-se **só a Entrada**, nunca um aviso retroativo "a aguardar reteste".
3. No histórico, entradas equivalentes são consolidadas; o preço de entrada, o stop e o alvo da primeira trade são mantidos.
4. Registo da intenção de envio **antes de chamar a API do Telegram**, para evitar repetição após reinício entre envio e gravação. Se Telegram falhar, **não há retry automático** daquele alerta; isto privilegia não repetir sobre entregar todos os alertas.
5. No **primeiro scan após arrancar com estado totalmente vazio**, os breakouts históricos são registados sem reenvio. Os sinais novos em scans seguintes podem ser enviados. Esta proteção evita enxurradas pós-deploy, mas pode perder um aviso que acabou de aparecer antes do primeiro scan.
6. **Upstash Redis REST (opcional)** guarda alertas, trades e estado de inicialização fora do Render. Usa um lock exclusivo para que duas instâncias ligadas à mesma base não corram o scanner simultaneamente. Sem Upstash funciona por JSON local, sujeito a perda após deploy.

## Render — comandos inalterados

Build: `pip install -r requirements.txt`

Start: `python start.py`

Não apagues `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `EXCHANGE_ID`, `MARKET_TYPE`, `TIMEFRAMES` nem os restantes parâmetros que já definiste.

Parâmetros adicionais opcionais:

```ini
SETUP_MERGE_BARS=16
SETUP_ZONE_TOLERANCE_PCT=0.018
```

## Proteção completa entre deploys sem custo — Upstash Free

O Render Free não garante persistência do filesystem. Para manter o histórico dos sinais e trades entre deployments, escolhe **Upstash Redis — plano Free ($0)**. O serviço tem limites e a oferta pode mudar; verifica o plano antes de criares.

1. Regista-te em https://console.upstash.com/ e cria uma base Redis no plano **Free**. Não escolhas Pay as You Go se queres evitar custos.
2. Nas opções da base, copia a **REST URL (HTTPS)** e o **REST Token**.
3. Em Render > Environment, cria **estas duas variáveis secretas**, mantendo os restantes parâmetros:

```ini
UPSTASH_REDIS_REST_URL=https://<teu-endpoint>.upstash.io
UPSTASH_REDIS_REST_TOKEN=<token-secreto>
```

4. Faz deploy da V8. O programa usa a mesma memória remota para proteger contra alertas duplicados e restaurar o JSON das trades em cada scan.

**Não partilhes** as credenciais REST nem as coloques no GitHub. Quando a Upstash não responde, o scanner suspende os envios em vez de arriscar duplicações. A primeira execução sem estado guarda a linha de base sem reenviar sinais antigos. Se duas instâncias executarem ao mesmo tempo, usa um bloqueio remoto temporário para evitar concorrência.

**Imagens:** as imagens PNG são ficheiros locais e não ficam no Redis; após deploy podem precisar de ser geradas novamente. O histórico *numérico* das trades (entrada/SL/TP/status) fica persistido na base remota. Os sinais não são ordens executadas, e o win rate é simulado com candles.

## Testes

```bash
python test_v5.py
python test_dedup_v8.py
```

Foram testados cenários artificiais, **não dados reais da MEXC nem um deploy live**. O segundo teste simula a duplicação INJ/USDT e seis scans, além de mock de persistência remota e exclusão mútua de scanners.

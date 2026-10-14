# Scanner RSI + Rompimentos V5 — Render gratuito

Arquitetura: **um único Render Web Service** com `start.py`, que inicia Streamlit (`app.py`) e o scanner periódico (`rsi_divergence_bot.py`). Não exige base de dados paga nem API privada da exchange.

## Instalação

1. Copiar os ficheiros deste pacote para a **raiz** do repositório GitHub, **sem renomear** `app.py`, `start.py`, `breakout_setup.py`, `divergence_engine.py`, `open_interest.py`, `charting.py` ou `rsi_divergence_bot.py`.
2. Render Web Service → **Build Command**: `pip install -r requirements.txt`.
3. **Start Command**: `python start.py`.
4. Configurar em Environment o token e chat ID do Telegram, sempre como variáveis, NUNCA no GitHub.
5. Fazer deploy e consultar os logs. Deixar o UptimeRobot a efetuar ping ao URL do painel (isto ajuda a evitar inatividade, mas não garante uptime 100%).

## Variáveis recomendadas

```env
EXCHANGE_ID=mexc
MARKET_TYPE=spot
SCAN_INTERVAL_SECONDS=300
SCAN_TOP_N=35
TIMEFRAMES=15m,30m,1h,4h,1d
OHLCV_LIMIT=320
TELEGRAM_BOT_TOKEN=TOKEN_PRIVADO
TELEGRAM_CHAT_ID=CHAT_ID_PRIVADO
EXTRA_SYMBOLS=ZECUSDT,STRKUSDT
OI_MODE=confirm_if_available
OI_SOURCE_EXCHANGE=binance
OI_MIN_DROP_PCT=2.0
MIN_RSI_DIFFERENCE=3.0
MIN_PRICE_DIFFERENCE_PCT=0.006
MIN_PRICE_DELTA_ATR=0.60
PRICE_PIVOT_PROMINENCE_ATR=0.9
REJECT_WICK_FRACTION=0.32
MIN_REJECTION_POINTS=2
MAX_SIGNAL_AGE_BARS=6
```

### Spot ou futuros

- `MARKET_TYPE=spot` procura apenas pares spot `ZEC/USDT`.
- `MARKET_TYPE=swap` procura apenas contratos perpétuos USDT margined `ZEC/USDT:USDT` na MEXC (quando suportados).
- Se o gráfico do TradingView for *PERPETUAL FUTURES*, usar `swap` para que a origem dos candles corresponda ao tipo de mercado. Não somar os dois mercados como se fossem a mesma série.
- `EXTRA_SYMBOLS=ZECUSDT` é resolvido automaticamente para a notação CCXT certa para o modo escolhido.

### Open Interest

- **Spot não tem OI.** O scanner utiliza OI histórico de **perpétuos** obtido via `ccxt.fetch_open_interest_history()`, por predefinição dos futuros da Binance, apenas para o *mesmo subjacente e timeframe*.
- A MEXC não está listada na documentação geral CCXT como suportada por `fetchOpenInterestHistory` unificado. Consequentemente, a configuração padrão usa a Binance como fonte externa, com indicação expressa de que é outro mercado/exchange.
- A verificação usa apenas medições históricas próximas das velas dos pivots, **não** o valor atual aplicado retroativamente a pivots antigos.
- `OI_MODE=confirm_if_available`: com OI histórico suficiente, requer queda mínima de 2% entre os dois últimos pivots; quando o OI está indisponível, o sinal RSI pode sair com a nota **"OI indisponível"** (não é tratado como confirmação).
- `OI_MODE=require`: não admite nenhum sinal RSI sem OI histórico suficiente e confirmatório. Muitos ativos da MEXC não estão listados na Binance, pelo que esta opção pode provocar *zero* alertas.
- `OI_MODE=off`: não solicita OI. É a opção para diagnosticar problemas com APIs de terceiros.
- OI a cair com preço a fazer novo extremo pode indicar menor participação alavancada, mas **não prova** reversão; OI crescente também não é necessariamente confirmação de continuação.

### Divergências RSI (muito seletivas)

- Timeframes: apenas `15m`, `30m`, `1h`, `4h`, `1d`. `2h`, `6h` e `12h` são ignorados mesmo que apareçam em `TIMEFRAMES`.
- `15m`, `30m`, `1h`: **mínimo três pivots estruturais consecutivos**. `4h`, `1d`: podem ser dois ou mais.
- Encontram-se primeiro máximos/mínimos **do preço**, por prominência e distância; RSI 14 Wilder é medido no fecho da **mesma vela**.
- Cada nova perna tem de superar os extremos de preço anteriores por pelo menos `max(0.6% do preço, 0.6×ATR)`, enquanto o RSI se desloca no sentido oposto pelo menos 3 pontos.
- Nenhum pivot estrutural intermédio pode ser omitido para fabricar uma divergência. Se o RSI do terceiro ponto quebra a sequência, **a sequência inteira é rejeitada**.
- Cada extremidade tem de ser relevante e o **último pivot deve exibir rejeição ou liquidity sweep**; pelo menos 2 pivots têm de apresentar essas características.
- Também se exige que o RSI da vela de extremo esteja próximo de um extremo local do indicador (`RSI_LOCAL_TOLERANCE=1.5`). Esta opção torna o scanner mais seletivo, podendo excluir padrões visualmente interessantes.
- Um novo máximo/mínimo trivial como o exemplo STRK deixa de cumprir o critério de percentagem/ATR.

### Breakout e trades

O motor de breakout da V4 continua a procurar **resistência com volume**, rompimento, sustentação em `4h/1d` e reteste. O alerta `Possível setup` contém gráfico verde/vermelho com a zona. O alerta `Entrada` contém entrada, stop abaixo da zona com buffer ATR, TP com **RR planeado 2:1**, gráfico e zonas de risco/recompensa.

Em `Trades e win rate` cada entrada simulada mostra uma miniatura do gráfico, **Em curso**, **Ganha** ou **Perdida**, entrada, stop, alvo, abertura e fecho. Os gráficos são gravados em `trade_charts/` e atualizados quando uma trade fecha.

**As trades são SIMULAÇÕES** inferidas a partir dos candles, não ordens executadas. O win rate não inclui comissões, slippage, execução real ou alterações de liquidez. A vela onde ocorre a entrada não é usada para simular o resultado futuro. Se um candle futuro alcançar SL e TP, assume-se primeiro SL (prudência sem dados intrabar).

## Atenção à persistência gratuita

Sem base de dados e com o disco **efémero** do Render Free, os JSON com histórico/alertas e imagens PNG são perdidos após alguns reinícios e novos deploys. O win rate apresentado é **apenas sobre as trades registadas desde a última reposição de dados**, não uma estatística permanente. Uma futura integração com armazenamento remoto gratuito poderá resolver isto.

## Testes locais

```bash
pip install -r requirements.txt
python test_v5.py
```

Os testes são **sintéticos e offline**. Não demonstram taxa de acerto no mercado real. Um teste com exchange simulada exercita o scanner completo, o painel usa o ficheiro gerado. Para uma validação real, executa `python rsi_divergence_bot.py --once` num ambiente com acesso à API da exchange. Compara os screenshots TradingView com candles do **mesmo mercado**, timeframe e feed de preço.

## Segurança

Se o token Telegram alguma vez foi incluído no código ou no histórico do GitHub, revoga-o no BotFather e substitui-o por outro. Não partilhes tokens nem `chat_id` em commits públicos.

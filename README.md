# Scanner V7 — correção de duplicados

Corrige o registo e os alertas repetidos do mesmo breakout, como o caso INJ/USDT 4h com a mesma entrada e stops ligeiramente diferentes.

## Alterações

- Identidade do sinal de rompimento passa a usar a vela do rompimento, não o preço flutuante da resistência.
- Matching semântico da zona: mesma exchange/mercado, par, timeframe, rompimento com diferença de até uma vela e níveis no máximo 1,2% diferentes. Setups temporalmente distintos continuam independentes.
- Apenas um alerta **Possível setup** e um **Entrada** por oportunidade, não dois de cada.
- O histórico antigo é deduplicado automaticamente quando o scanner arranca. Conserva a primeira sugestão de SL/TP; se uma das linhas duplicadas já estiver fechada, conserva o resultado final.
- IDs de alertas da V6 são interpretados para evitar reenvios na migração, desde que os ficheiros JSON locais ainda existam.
- Critérios RSI, OI, geração de gráficos leve e monitorização de trades da V6 foram mantidos.

## Render

Build: `pip install -r requirements.txt`  
Start: `python start.py`

Substituir todos os ficheiros da V6 pelos do pacote V7 no mesmo diretório do GitHub. Mantém as mesmas Environment Variables.

## Aviso sobre o Render gratuito

`trades_history.json`, `sent_alerts.json` e `alert_history.json` são ficheiros locais efémeros. Um **deploy** pode repor o sistema de ficheiros e apagar o estado anterior. Se isto acontecer, nenhuma solução baseada apenas nesses ficheiros pode garantir deduplicação entre deploys. Para garantia total sem pagar pela base de dados é preciso uma persistência externa (por exemplo, escrever um ledger através da API de outro serviço ou separar o scanner e guardar snapshots num repositório privado com credencial restrita). Não coloques tokens num repositório público.

## Testes

`python test_v5.py` e `python test_dedup_v7.py`

from dedup_setup import same_setup, deduplicate_trades, deduplicate_signals, already_traded
from rsi_divergence_bot import upsert_trade

BASE = 1_760_000_000_000

def signal(level=7.81, breakout=BASE, kind='Entrada', sym='INJ/USDT'):
    return {
        'id': f'mexc_INJUSDT_breakout_4h_{breakout}_{level}_{"entry" if kind == "Entrada" else "possible"}',
        'setup_id': f'breakout_4h_{breakout}_{level}',
        'categoria': 'breakout', 'tipo': 'bullish', 'fase':kind,
        'symbol':sym, 'timeframe':'4h', 'signal_time_ms':BASE + 3*14_400_000,
        'nivel':level, 'zona_min': level-.07, 'zona_max': level+.07,
        'trade':{'entry':7.952,'sl':7.59,'tp':8.676,'rr':2.0},
        'signal_time':'2026-10-14 08:00 UTC'
    }

def test():
    a = signal()
    b = signal(7.84)
    c = signal(8.35)
    d = signal(7.84, BASE + 10*14_400_000)
    assert same_setup(a,b)
    assert not same_setup(a,c)
    assert not same_setup(a,d)
    assert not same_setup(a, signal(sym='BTC/USDT'))
    assert len(deduplicate_signals([a,b])) == 1
    assert len(deduplicate_signals([a,signal(kind='Possível setup')])) == 2
    trades=[]
    upsert_trade(trades,a)
    upsert_trade(trades,b)
    assert len(trades)==1
    assert trades[0]['sl']==7.59
    assert already_traded(b,trades)
    legacy_b = dict(trades[0],trade_id=b['id'], setup_id=b['setup_id'], sl=7.60, tp=8.6552)
    fixed, num=deduplicate_trades([trades[0],legacy_b])
    assert len(fixed)==1 and num==1 and fixed[0]['sl']==7.59
    upsert_trade(fixed,c)
    assert len(fixed)==2
    print('OK: identidade semântica, alertas por fase, imports V6, histórico V6 e oportunidades independentes')

if __name__=='__main__': test()

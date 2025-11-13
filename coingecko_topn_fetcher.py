import os
import time
import requests
import pandas as pd


BASE_URL_DEMO = "https://api.coingecko.com/api/v3"
BASE_URL_PRO = "https://pro-api.coingecko.com/api/v3"


def _build_base_url(plan: str) -> str:
  plan = plan.lower()
  if plan == "demo":
    return BASE_URL_DEMO
  elif plan == "pro":
    return BASE_URL_PRO
  else:
    raise ValueError("plan must be 'demo' or 'pro'")


def _build_auth_params(api_key: str | None, plan: str) -> dict:
  if not api_key:
    return {}

  plan = plan.lower()
  if plan == "demo":
    return {"x_cg_demo_api_key": api_key}
  elif plan == "pro":
    return {"x_cg_pro_api_key": api_key}
  else:
    raise ValueError("plan must be 'demo' or 'pro'")


def fetch_top_coins_current_mc(
  vs_currency: str = "usd",
  per_page: int = 250,
  page: int = 1,
  order: str = "market_cap_desc",
  api_key: str | None = None,
  plan: str = "demo",
) -> list[dict]:
  base_url = _build_base_url(plan)
  url = f"{base_url}/coins/markets"

  params = {
    "vs_currency": vs_currency,
    "order": order,
    "per_page": per_page,
    "page": page,
    "sparkline": "false",
  }
  params.update(_build_auth_params(api_key, plan))

  resp = requests.get(url, params=params)
  if resp.status_code == 429:
    raise RuntimeError("Hit CoinGecko rate limit (HTTP 429) in /coins/markets.")
  resp.raise_for_status()
  return resp.json()


def fetch_market_caps_series(
  coin_id: str,
  vs_currency: str = "usd",
  window_days: int = 14,
  api_key: str | None = None,
  plan: str = "demo",
  interval: str = "daily",
) -> pd.Series:
  base_url = _build_base_url(plan)
  url = f"{base_url}/coins/{coin_id}/market_chart"

  params = {
    "vs_currency": vs_currency,
    "days": str(window_days),
    "interval": interval,
  }
  params.update(_build_auth_params(api_key, plan))

  resp = requests.get(url, params=params)
  if resp.status_code == 429:
    raise RuntimeError(f"Hit CoinGecko rate limit (HTTP 429) for {coin_id} in /market_chart.")
  resp.raise_for_status()
  data = resp.json()

  mc = data.get("market_caps", [])
  dates = []
  caps = []

  for ts, cap in mc:
    dt = pd.to_datetime(ts, unit="ms", utc=True)
    dates.append(dt.strftime("%Y-%m-%d"))
    caps.append(cap)

  series = pd.Series(data=caps, index=dates)
  return series


def select_top_n_by_avg_market_cap(
  top_n: int = 50,
  window_days: int = 14,
  vs_currency: str = "usd",
  api_key: str | None = None,
  plan: str = "demo",
  candidate_multiplier: int = 3,
  sleep_sec: float = 1.2,
) -> list[dict]:

  stable_symbols = {
    "usdt", "usdc", "busd", "dai", "tusd",
    "usdd", "usdp", "gusd", "lusd", "susd", "fdusd"
  }

  per_page = min(top_n * candidate_multiplier, 250)
  raw_coins = fetch_top_coins_current_mc(
    vs_currency=vs_currency,
    per_page=per_page,
    page=1,
    api_key=api_key,
    plan=plan,
  )

  print(f"[INFO] Fetched {len(raw_coins)} candidate coins from /coins/markets")

  results = []

  for coin in raw_coins:
    coin_id = coin["id"]
    symbol = coin["symbol"].lower()

    # 스테이블 코인 필터링
    if symbol in stable_symbols:
      print(f"[SKIP] {coin_id} ({symbol}): stablecoin")
      continue

    try:
      series = fetch_market_caps_series(
        coin_id=coin_id,
        vs_currency=vs_currency,
        window_days=window_days,
        api_key=api_key,
        plan=plan,
        interval="daily",
      )

      non_null_count = series.count()
      if non_null_count < window_days:
        print(f"[SKIP] {coin_id}: only {non_null_count} non-null days (< {window_days})")
        time.sleep(sleep_sec)
        continue

      avg_mc = float(series.mean())

      binance_symbol = symbol.upper() + "USDT"

      results.append({
        "id": coin_id,
        "symbol": symbol,
        "binance_symbol": binance_symbol,
        "avg_market_cap": avg_mc,
      })

      print(f"[OK] {coin_id} -> {binance_symbol}: avg_mc={avg_mc:.2f}")

      time.sleep(sleep_sec)

    except RuntimeError as e:
      print(f"[WARN] {coin_id}: runtime error -> {e}")
      break
    except Exception as e:
      print(f"[WARN] {coin_id}: error -> {e}")
      time.sleep(sleep_sec)
      continue

  results_sorted = sorted(results, key=lambda x: x["avg_market_cap"], reverse=True)
  return results_sorted[:top_n]


if __name__ == "__main__":
  api_key = "CG-Ghdp828k7P65jnEh1kUniw99"

  top_n = 30
  window_days = 14

  top_tokens = select_top_n_by_avg_market_cap(
    top_n=top_n,
    window_days=window_days,
    vs_currency="usd",
    api_key=api_key,
    plan="demo",         # 또는 "pro"
    candidate_multiplier=3,
    sleep_sec=1.2,
  )

  print("\n=== FINAL TOP TOKENS BY AVG MARKET CAP (BINANCE FORMAT) ===")
  for i, token in enumerate(top_tokens, start=1):
    print(f"{i:2d}. {token['binance_symbol']}: "
          f"avg_mc={token['avg_market_cap']:.2f}  (id={token['id']})")
    
# failed due to bad results, too much noise.staked eth, binancebnb, weth, wbtc..
  """ results
  === FINAL TOP TOKENS BY AVG MARKET CAP (BINANCE FORMAT) ===
 1. BTCUSDT: avg_mc=2093654775315.86  (id=bitcoin)
 2. ETHUSDT: avg_mc=428697951599.14  (id=ethereum)
 3. XRPUSDT: avg_mc=143501191213.32  (id=ripple)
 4. BNBUSDT: avg_mc=137948712752.43  (id=binancecoin)
 5. SOLUSDT: avg_mc=92065038900.85  (id=solana)
 6. STETHUSDT: avg_mc=30602475296.48  (id=staked-ether)
 7. TRXUSDT: avg_mc=27665476403.08  (id=tron)
 8. DOGEUSDT: avg_mc=26642417866.91  (id=dogecoin)
 9. ADAUSDT: avg_mc=20886904117.40  (id=cardano)
10. WSTETHUSDT: avg_mc=14310898933.58  (id=wrapped-steth)
11. WBTCUSDT: avg_mc=13297348139.69  (id=wrapped-bitcoin)
12. FIGR_HELOCUSDT: avg_mc=13150178493.86  (id=figure-heloc)
13. WBETHUSDT: avg_mc=12541143622.72  (id=wrapped-beacon-eth)
14. HYPEUSDT: avg_mc=11106108061.71  (id=hyperliquid)
15. LINKUSDT: avg_mc=11060013447.46  (id=chainlink)
16. BCHUSDT: avg_mc=10214627028.41  (id=bitcoin-cash)
17. WBTUSDT: avg_mc=9830604737.45  (id=whitebit)
18. WEETHUSDT: avg_mc=9376585943.67  (id=wrapped-eeth)
19. XLMUSDT: avg_mc=9234801790.27  (id=stellar)
20. USDSUSDT: avg_mc=9229543600.17  (id=usds)
21. BSC-USDUSDT: avg_mc=8982568435.07  (id=binance-bridged-usdt-bnb-smart-chain)
22. USDEUSDT: avg_mc=8828662551.81  (id=ethena-usde)
23. LEOUSDT: avg_mc=8598788215.58  (id=leo-token)
24. ZECUSDT: avg_mc=7977693362.16  (id=zcash)
25. WETHUSDT: avg_mc=7923189522.20  (id=weth)
26. SUIUSDT: avg_mc=7836631399.71  (id=sui)
27. HBARUSDT: avg_mc=7705611768.79  (id=hedera-hashgraph)
28. CBBTCUSDT: avg_mc=7628915787.65  (id=coinbase-wrapped-btc)
29. AVAXUSDT: avg_mc=7441156847.51  (id=avalanche-2)
30. LTCUSDT: avg_mc=7435741376.00  (id=litecoin)
  """

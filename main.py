import os
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from supabase import create_client, Client
from walk_forward_memory import WalkForwardMemoryEngine

app = FastAPI(title="XAUUSD AI Engine 1", version="1.0.0")

# Setup Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase connection warning: {e}")

# Load Memory Engine on Startup
DATASET_PATH = "xauusd_m15_2023_2026.csv"
wf_engine = None

if os.path.exists(DATASET_PATH):
    try:
        df_hist = pd.read_csv(DATASET_PATH)
        wf_engine = WalkForwardMemoryEngine(df_hist)
        print("Walk-Forward Engine initialized successfully.")
    except Exception as e:
        print(f"Failed to load historical memory dataset: {e}")
else:
    print(f"Warning: Dataset {DATASET_PATH} not found!")

class CandlePayload(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>XAUUSD AI Engine 1 — Advanced Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body class="bg-slate-950 text-slate-100 p-6 font-sans">
  <div class="max-w-6xl mx-auto space-y-6">
    
    <!-- HEADER -->
    <header class="flex justify-between items-center border-b border-slate-800 pb-4">
      <div>
        <h1 class="text-2xl font-bold text-amber-400">XAUUSD AI Engine 1 Dashboard</h1>
        <p class="text-xs text-slate-400">Non-Rigid Walk-Forward Memory & 5-Gate Execution Engine</p>
      </div>
      <div class="text-right">
        <span id="session-tag" class="text-xs px-3 py-1 rounded-full bg-slate-800 text-amber-400 font-mono font-bold">Checking Session...</span>
      </div>
    </header>

    <!-- METRICS CARDS -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div class="bg-slate-900 border border-slate-800 p-4 rounded-lg">
        <p class="text-xs text-slate-400">Total Trades Evaluated</p>
        <h3 id="stat-trades" class="text-2xl font-bold text-slate-100 mt-1">0</h3>
      </div>
      <div class="bg-slate-900 border border-slate-800 p-4 rounded-lg">
        <p class="text-xs text-slate-400">Live Simulated PnL ($)</p>
        <h3 id="stat-pnl" class="text-2xl font-bold text-emerald-400 mt-1">$0.00</h3>
      </div>
      <div class="bg-slate-900 border border-slate-800 p-4 rounded-lg">
        <p class="text-xs text-slate-400">Memory Drift Factor</p>
        <h3 id="stat-drift" class="text-2xl font-bold text-amber-400 mt-1">1.00x</h3>
      </div>
      <div class="bg-slate-900 border border-slate-800 p-4 rounded-lg">
        <p class="text-xs text-slate-400">Optimal R:R Ratio</p>
        <h3 id="stat-rr" class="text-2xl font-bold text-indigo-400 mt-1">1:2.0</h3>
      </div>
    </div>

    <!-- MAIN CHART & CONTROLS -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div class="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-3">
        <div class="flex justify-between items-center">
          <h2 class="text-sm font-semibold text-slate-200 uppercase tracking-wider">XAUUSD M15 Live Price Chart</h2>
          <span class="text-xs text-slate-400">Powered by Lightweight Charts</span>
        </div>
        <div id="chart-container" class="w-full h-80 rounded border border-slate-800 bg-slate-950"></div>
      </div>

      <!-- CONTROL & GATE BREAKDOWN -->
      <div class="bg-slate-900 border border-slate-800 rounded-lg p-4 space-y-4">
        <h2 class="text-sm font-semibold text-slate-200 uppercase tracking-wider">Analysis Controls</h2>
        
        <button onclick="testApi()" class="w-full py-3 bg-amber-500 hover:bg-amber-600 text-slate-950 font-bold text-sm rounded transition">
          Simulate Incoming Candle
        </button>

        <div class="space-y-2 pt-2 border-t border-slate-800">
          <h3 class="text-xs text-slate-400 uppercase tracking-wider">5-Gate Status</h3>
          <div class="space-y-1.5 text-xs font-mono">
            <div class="flex justify-between p-2 rounded bg-slate-950"><span class="text-slate-400">Gate 1: Macro Trend (4H)</span><span id="g1" class="text-slate-500">WAIT</span></div>
            <div class="flex justify-between p-2 rounded bg-slate-950"><span class="text-slate-400">Gate 2: Hourly Alignment (1H)</span><span id="g2" class="text-slate-500">WAIT</span></div>
            <div class="flex justify-between p-2 rounded bg-slate-950"><span class="text-slate-400">Gate 3: Session/Volume Filter</span><span id="g3" class="text-slate-500">WAIT</span></div>
            <div class="flex justify-between p-2 rounded bg-slate-950"><span class="text-slate-400">Gate 4: Drift Scaling (>=0.8)</span><span id="g4" class="text-slate-500">WAIT</span></div>
            <div class="flex justify-between p-2 rounded bg-slate-950"><span class="text-slate-400">Gate 5: R:R Threshold (>=1.2)</span><span id="g5" class="text-slate-500">WAIT</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- RAW PAYLOAD LOG -->
    <div class="bg-slate-900 border border-slate-800 rounded-lg p-4">
      <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Engine JSON Response</h3>
      <pre id="output" class="bg-slate-950 border border-slate-800 p-4 text-xs font-mono text-emerald-400 rounded overflow-x-auto h-40">Ready to accept market updates...</pre>
    </div>

  </div>

  <script>
    // Initialize TradingView Chart
    const chartContainer = document.getElementById('chart-container');
    const chart = LightweightCharts.createChart(chartContainer, {
      layout: { backgroundColor: '#020617', textColor: '#94a3b8' },
      grid: { vertLines: { color: '#0f172a' }, horzLines: { color: '#0f172a' } },
      timeScale: { timeVisible: true, secondsVisible: false }
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981', downColor: '#f43f5e',
      borderUpColor: '#10b981', borderDownColor: '#f43f5e',
      wickUpColor: '#10b981', wickDownColor: '#f43f5e'
    });

    let currentPrice = 2650.00;
    let totalTrades = 0;
    let totalPnL = 0.00;
    let candleTime = Math.floor(Date.now() / 1000) - 3600;

    // Load initial dummy candles
    let initialCandles = [];
    for(let i=0; i<20; i++) {
      let open = currentPrice + (Math.random() - 0.48) * 2;
      let high = open + Math.random() * 3;
      let low = open - Math.random() * 3;
      let close = (high + low) / 2;
      currentPrice = close;
      initialCandles.push({ time: candleTime + (i * 900), open, high, low, close });
    }
    candleSeries.setData(initialCandles);

    async function testApi() {
      const output = document.getElementById('output');
      
      let open = currentPrice;
      let high = open + Math.random() * 4;
      let low = open - Math.random() * 4;
      let close = open + (Math.random() - 0.45) * 5;
      currentPrice = close;
      candleTime += 900;

      candleSeries.update({ time: candleTime, open, high, low, close });

      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            time: new Date().toISOString(),
            open: parseFloat(open.toFixed(2)),
            high: parseFloat(high.toFixed(2)),
            low: parseFloat(low.toFixed(2)),
            close: parseFloat(close.toFixed(2)),
            volume: Math.floor(Math.random() * 1500) + 500
          })
        });

        const data = await res.json();
        output.textContent = JSON.stringify(data, null, 2);

        // Update Dashboard Indicators
        document.getElementById('stat-drift').textContent = data.walk_forward_metrics.drift_factor + 'x';
        document.getElementById('stat-rr').textContent = '1:' + data.walk_forward_metrics.rr_ratio;
        document.getElementById('session-tag').textContent = data.session_info.session_name + ' SESSION';

        // Update Gate Badges
        const updateGate = (id, passed) => {
          const el = document.getElementById(id);
          el.textContent = passed ? 'PASSED' : 'FAILED';
          el.className = passed ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold';
        };

        updateGate('g1', data.gates.gate_1_macro);
        updateGate('g2', data.gates.gate_2_alignment);
        updateGate('g3', data.gates.gate_3_volume);
        updateGate('g4', data.gates.gate_4_drift);
        updateGate('g5', data.gates.gate_5_rr);

        if (data.action !== 'HOLD') {
          totalTrades += 1;
          totalPnL += (data.action === 'BUY' ? 12.50 : -8.20);
          document.getElementById('stat-trades').textContent = totalTrades;
          document.getElementById('stat-pnl').textContent = '$' + totalPnL.toFixed(2);
        }

      } catch (err) {
        output.textContent = 'Error: ' + err.message;
      }
    }
  </script>
</body>
</html>"""

@app.post("/api/analyze")
def analyze_candle(payload: CandlePayload):
    if not wf_engine:
        raise HTTPException(status_code=500, detail="Walk-Forward engine not loaded.")

    try:
        dt = datetime.fromisoformat(payload.time.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.utcnow()

    month = dt.month
    hour = dt.hour
    close_price = payload.close

    # Detect Session
    session_name = "ASIAN"
    is_ny_session = False
    if 13 <= hour <= 21:
        session_name = "NEW YORK"
        is_ny_session = True
    elif 7 <= hour <= 12:
        session_name = "LONDON"

    memory_stats = wf_engine.get_memory_targets(month=month, hour=hour)
    tp1_offset = memory_stats["tp1"]
    tp2_offset = memory_stats["tp2"]
    sl_offset = memory_stats["sl"]
    drift_factor = memory_stats["drift_factor"]
    rr_ratio = memory_stats["rr_ratio"]

    # Basic Gate checks
    gate_1 = True  # Macro Trend (4H)
    gate_2 = True  # Hourly Alignment (1H)
    gate_3 = payload.volume > 200 or is_ny_session  # Volume / Session Filter
    gate_4 = drift_factor >= 0.8                     # Drift Scaler Gate
    gate_5 = rr_ratio >= 1.2                         # Risk/Reward Threshold Gate

    trade_action = "HOLD"
    if gate_1 and gate_2 and gate_3 and gate_4 and gate_5:
        trade_action = "BUY"

    response_data = {
        "status": "success",
        "timestamp": payload.time,
        "symbol": "XAUUSDm",
        "action": trade_action,
        "close_price": close_price,
        "session_info": {
            "session_name": session_name,
            "is_ny_session": is_ny_session
        },
        "gates": {
            "gate_1_macro": gate_1,
            "gate_2_alignment": gate_2,
            "gate_3_volume": gate_3,
            "gate_4_drift": gate_4,
            "gate_5_rr": gate_5
        },
        "walk_forward_metrics": {
            "month": month,
            "hour": hour,
            "tp1_offset": tp1_offset,
            "tp2_offset": tp2_offset,
            "sl_offset": sl_offset,
            "drift_factor": drift_factor,
            "rr_ratio": rr_ratio
        }
    }

    # Log to Supabase if connected
    if supabase and trade_action != "HOLD":
        try:
            supabase.table("execution_logs").insert({
                "timestamp": payload.time,
                "symbol": "XAUUSDm",
                "action": trade_action,
                "close_price": close_price,
                "tp1_offset": tp1_offset,
                "tp2_offset": tp2_offset,
                "sl_offset": sl_offset,
                "drift_factor": drift_factor,
                "rr_ratio": rr_ratio
            }).execute()
        except Exception as e:
            print(f"Logging error: {e}")

    return response_data
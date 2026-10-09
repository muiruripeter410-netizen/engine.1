BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "xauusd_m15_2023_2026.csv")
wf_engine = None

if os.path.exists(DATASET_PATH):
    try:
        df_hist = pd.read_csv(DATASET_PATH)
        wf_engine = WalkForwardMemoryEngine(df_hist)
        print(f"Walk-Forward Engine initialized using dataset at {DATASET_PATH}.")
    except Exception as e:
        print(f"Failed to parse dataset at {DATASET_PATH}: {e}")
else:
    print(f"Warning: Dataset not found at {DATASET_PATH}! Using dynamic fallback memory metrics.")
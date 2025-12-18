import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_HISTORY_PATH = Path(".cache/ticker_history.json")
DEFAULT_METRICS = ["buy_score", "price"]
VALID_METRICS = {"buy_score", "price", "pe_ratio", "revenue_growth", "risk_reward"}


def load_history(path: Path) -> dict:
    """Carica lo storico dei ticker dal file JSON."""
    if not path.exists():
        print(f"Nessuno storico trovato in {path}. Esegui prima stockAnalyzer.py.")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Errore nel leggere lo storico: {exc}")
        sys.exit(1)


def build_dfs(history: dict, tickers: list) -> dict:
    """Crea un DataFrame per ogni ticker richiesto, ordinato per data."""
    frames = {}
    for ticker in tickers:
        entries = history.get(ticker, [])
        if not entries:
            continue
        df = pd.DataFrame(entries)
        if df.empty:
            continue
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        frames[ticker] = df
    return frames


def filter_frames_by_top_metric(frames: dict, top_n: int | None, metric: str) -> dict:
    """Restituisce solo i primi N ticker in base all'ultimo valore di una metrica."""
    if not top_n or top_n <= 0:
        return frames

    scored = []
    for ticker, df in frames.items():
        if metric not in df.columns:
            continue
        series = df[metric].dropna()
        if series.empty:
            continue
        scored.append((ticker, series.iloc[-1]))

    if not scored:
        print(
            f"Nessun dato per applicare il filtro top con metrica '{metric}'. Mostro tutti i ticker."
        )
        return frames

    scored.sort(key=lambda item: item[1], reverse=True)
    selected = {ticker for ticker, _ in scored[:top_n]}
    filtered = {ticker: frames[ticker] for ticker in frames if ticker in selected}

    if filtered and len(filtered) != len(frames):
        names = ", ".join(sorted(filtered))
        print(f"Filtro top attivo ({metric}): visualizzo {names}")

    return filtered


def plot_history(frames: dict, metrics: list, save_path: str | None = None):
    """Disegna l'andamento delle metriche per i ticker selezionati."""
    if not frames:
        print("Nessun dato valido da visualizzare.")
        sys.exit(0)

    # Calcola l'intervallo temporale globale per allineare tutti i grafici.
    ts_series = [df["timestamp"] for df in frames.values() if "timestamp" in df.columns]
    all_timestamps = (
        pd.concat(ts_series, ignore_index=True)
        if ts_series
        else pd.Series(dtype="datetime64[ns]")
    )

    rows = len(metrics)
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(11, 4 * rows))
    # Lascia spazio a destra per una legenda esterna.
    fig.subplots_adjust(right=0.82)

    if rows == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        has_data = False
        for ticker, df in frames.items():
            if metric not in df.columns:
                continue
            (line,) = ax.plot(df["timestamp"], df[metric], label=ticker, linewidth=2)
            # Marker piccoli per vedere chiaramente gli snapshot nel tempo.
            ax.scatter(
                df["timestamp"],
                df[metric],
                s=12,
                alpha=0.55,
                color=line.get_color(),
                label="_nolegend_",
            )

            median_val = df[metric].median()
            if pd.notna(median_val):
                ax.axhline(
                    median_val,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.4,
                    color=line.get_color(),
                    label="_nolegend_",
                )
            has_data = True

        ax.set_title(f"Andamento {metric}", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        # Allinea gli assi temporali al range completo disponibile.
        if not all_timestamps.empty:
            ax.set_xlim(all_timestamps.min(), all_timestamps.max())
        if has_data:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=9,
                    frameon=True,
                    borderaxespad=0.6,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Nessun dato per questo indicatore",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"Grafico salvato in {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualizza l'andamento storico dei ticker salvato in .cache/ticker_history.json"
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Ticker da tracciare (default: tutti quelli presenti nello storico).",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help=(
            "Metriche da tracciare separate da virgola. "
            "Opzioni: buy_score, price, pe_ratio, revenue_growth, risk_reward"
        ),
    )
    parser.add_argument(
        "--history-path",
        default=str(DEFAULT_HISTORY_PATH),
        help="Percorso del file di storico (default: .cache/ticker_history.json).",
    )
    parser.add_argument(
        "--save",
        help="Percorso file immagine per salvare il grafico (es. output.png). "
        "Se non impostato, mostra il grafico a schermo.",
    )
    parser.add_argument(
        "--top",
        type=int,
        metavar="N",
        help="Mostra solo i primi N ticker ordinati per l'ultima metrica indicata in --top-metric.",
    )
    parser.add_argument(
        "--top-metric",
        default="buy_score",
        choices=sorted(VALID_METRICS),
        help="Metrica da usare per il filtro top (default: buy_score).",
    )
    args = parser.parse_args()

    history_path = Path(args.history_path)
    history = load_history(history_path)

    available = sorted(history.keys())
    if args.tickers:
        requested = [t.upper() for t in args.tickers]
        tickers = [t for t in requested if t in history]
        missing = [t for t in requested if t not in history]
        if missing:
            print(f"Attenzione: nessuno storico per {', '.join(missing)}")
    else:
        tickers = available

    if not tickers:
        print("Nessun ticker con storico da mostrare.")
        sys.exit(0)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metrics = [m for m in metrics if m in VALID_METRICS]
    if not metrics:
        metrics = DEFAULT_METRICS

    frames = build_dfs(history, tickers)
    frames = filter_frames_by_top_metric(frames, args.top, args.top_metric)
    plot_history(frames, metrics, save_path=args.save)


if __name__ == "__main__":
    main()
"""
Batch extraction of trend strength data.
"""
import copy
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from json import JSONEncoder
from math import isnan
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datetime import datetime, date

from trendvisdata.chart_data import Data
from trendvisdata.market_data import MktUtils, NorgateExtract, YahooExtract
from trendvisdata.sector_mappings import sectmap
from trendvisdata.trend import TrendStrength
from trendvisdata.trend_params import trend_params_dict  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

class NumpyDateEncoder(json.JSONEncoder):
    """
    JSON encoder for numpy scalar types, pandas Series/DataFrame, and
    Python datetime objects.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 2)
        if isinstance(obj, float):
            return round(obj, 2)
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, pd.DatetimeIndex):
            return obj.date.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_json()
        return super().default(obj)


def _nan_to_none(obj: Any) -> Any:
    """Recursively replace float NaN values with None."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and isnan(obj):
        return None
    return obj


class _NanConverter(JSONEncoder):
    """JSON encoder replacing NaN with null."""

    def encode(self, obj: Any, *args: Any, **kwargs: Any) -> str:
        return super().encode(_nan_to_none(obj), *args, **kwargs)


# ---------------------------------------------------------------------------
# Module-level worker function
# ---------------------------------------------------------------------------

def _execute_run(
    label: str,
    days: int,
    source: str,
    mkts: int,
    source_metadata: dict[str, Any],
    full_price_history: dict[str, pd.DataFrame],
    source_mappings: dict[str, Any],
    output_dir: Path,
) -> str:
    """
    Execute the trend strength pipeline for a single lookback window and
    write the output JSON file.

    Parameters
    ----------
    label : str
        Run label used as the output filename stem.
    days : int
        Lookback window in trading days.
    source : str
        Data source; ``'yahoo'`` or ``'norgate'``.
    mkts : int
        Number of markets for bar and market chart panels.
    source_metadata : dict[str, Any]
        Ticker list and name mappings from the shared fetch.
    full_price_history : dict[str, pd.DataFrame]
        Full price history from the max-lookback fetch.
    source_mappings : dict[str, Any]
        Sector mappings; deep-copied before use.
    output_dir : Path
        Directory to which the output JSON file is written.

    Returns
    -------
    str
        Label of the completed run.
    """
    params: dict[str, Any] = Data._init_params({
        'mkts': mkts,
        'source': source,
        'days': days,
        'lookback': days,
        'norm': False,
    })
    params['tickers'] = source_metadata['tickers']
    params['ticker_name_dict'] = source_metadata['ticker_name_dict']
    params['ticker_short_name_dict'] = source_metadata['ticker_short_name_dict']
    params['asset_type'] = 'Equity' if source == 'yahoo' else 'CTA'

    if source == 'norgate':
        params['init_ticker_dict'] = source_metadata['init_ticker_dict']

    params = MktUtils.date_set(params)

    start_dt: pd.Timestamp = pd.to_datetime(params['start_date'])
    window_ticker_dict: dict[str, pd.DataFrame] = {}

    for ticker, frame in full_price_history.items():
        ticker_price_history: pd.DataFrame = frame.loc[
            frame.index >= start_dt
        ].copy()
        if len(ticker_price_history) > 0:
            window_ticker_dict[ticker] = ticker_price_history
            params = MktUtils.window_set(frame=ticker_price_history, params=params)

    tables: dict[str, Any] = {'raw_ticker_dict': window_ticker_dict}
    run_mappings: dict[str, Any] = copy.deepcopy(source_mappings)

    tables = MktUtils.ticker_clean(params=params, tables=tables)
    tables = TrendStrength.trend_calc(
        params=params, tables=tables, mappings=run_mappings
    )
    _, tables = TrendStrength.top_trend_tickers(params=params, tables=tables)

    data_dict: dict[str, Any] = Data.get_all_data(params=params, tables=tables)

    output: dict[str, Any] = {'data_dict': data_dict}
    numpy_encoded: str = json.dumps(output, cls=NumpyDateEncoder)
    resolved_output: Any = json.loads(numpy_encoded)
    nan_encoded: str = json.dumps(resolved_output, cls=_NanConverter)
    serialised_output: Any = json.loads(nan_encoded)

    file_path: Path = output_dir / f"{label}.json"
    with open(file_path, 'w') as fp:
        json.dump(serialised_output, fp)

    return label


# ---------------------------------------------------------------------------
# Shared data fetch methods
# ---------------------------------------------------------------------------

class _DataFetch:
    """
    Static data fetch methods shared by ``TrendBatch`` and
    ``TrendBatchParallel``.
    """

    @staticmethod
    def yahoo(
        mkts: int,
        max_lookback: int,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
        """
        Retrieve the S&P 500 constituent list and price history.

        Parameters
        ----------
        mkts : int
            Number of markets for bar and market chart panels.
        max_lookback : int
            Number of trading days of history to retrieve.

        Returns
        -------
        source_metadata : dict[str, Any]
            Ticker list and name mappings.
        full_price_history : dict[str, pd.DataFrame]
            Raw OHLCV DataFrames keyed by ticker symbol.
        source_mappings : dict[str, Any]
            Sector mappings including ``sector_mappings_df``.
        """
        params: dict[str, Any] = Data._init_params({
            'mkts': mkts,
            'source': 'yahoo',
            'days': max_lookback,
            'lookback': max_lookback,
            'norm': False,
        })
        source_mappings: dict[str, Any] = copy.deepcopy(sectmap)

        params, source_mappings = YahooExtract.ticker_extract(
            params=params, mappings=source_mappings
        )
        params['ticker_short_name_dict'] = params['ticker_name_dict']
        params['asset_type'] = 'Equity'
        params = MktUtils.date_set(params)

        tables: dict[str, Any] = {}
        params, tables = YahooExtract.import_yahoo(params=params, tables=tables)

        source_metadata: dict[str, Any] = {
            'tickers': params['tickers'],
            'ticker_name_dict': params['ticker_name_dict'],
            'ticker_short_name_dict': params['ticker_short_name_dict'],
        }

        return source_metadata, tables['raw_ticker_dict'], source_mappings

    @staticmethod
    def norgate(
        mkts: int,
        max_lookback: int,
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
        """
        Retrieve the Norgate ticker universe and price history.

        Parameters
        ----------
        mkts : int
            Number of markets for bar and market chart panels.
        max_lookback : int
            Number of trading days of history to retrieve.

        Returns
        -------
        source_metadata : dict[str, Any]
            Ticker list, name mappings, and init_ticker_dict.
        full_price_history : dict[str, pd.DataFrame]
            Raw OHLCV DataFrames keyed by ticker symbol.
        source_mappings : dict[str, Any]
            Sector mappings including ``sector_mappings_df``.
        """
        params: dict[str, Any] = Data._init_params({
            'mkts': mkts,
            'source': 'norgate',
            'days': max_lookback,
            'lookback': max_lookback,
            'norm': False,
        })
        params['asset_type'] = 'CTA'
        source_mappings: dict[str, Any] = copy.deepcopy(sectmap)

        params = NorgateExtract.get_norgate_tickers(params=params)
        params = MktUtils.date_set(params)

        tables: dict[str, Any] = {}
        params, tables, source_mappings = NorgateExtract.import_norgate(
            params=params, tables=tables, mappings=source_mappings
        )

        source_metadata: dict[str, Any] = {
            'tickers': params['tickers'],
            'ticker_name_dict': params['ticker_name_dict'],
            'ticker_short_name_dict': params['ticker_short_name_dict'],
            'init_ticker_dict': params['init_ticker_dict'],
        }

        return source_metadata, tables['raw_ticker_dict'], source_mappings


# ---------------------------------------------------------------------------
# Sequential batch
# ---------------------------------------------------------------------------

class TrendBatch:
    """
    Execute the trend strength pipeline for a labelled set of runs.

    Price history is fetched once per source at the maximum lookback
    across that source's runs. Each run executes the full pipeline
    independently with its own params, ticker_clean, trend_calc, and
    serialisation.

    Parameters
    ----------
    batch_config : dict[str, dict[str, str | int]]
        Mapping of output label to run parameters. Each entry must
        contain ``'days'`` (int) and ``'source'`` (str: ``'yahoo'`` or
        ``'norgate'``).
    cob_date : str
        Close-of-business date in YYYYMMDD format; used as the output
        subfolder name.
    output_root : Path
        Root directory under which the cob_date subfolder is created.
    mkts : int
        Number of markets for bar and market chart panels, applied
        uniformly across all runs.
    """

    def __init__(
        self,
        batch_config: dict[str, dict[str, str | int]],
        cob_date: str,
        output_root: Path,
        mkts: int,
    ) -> None:
        self._output_dir: Path = output_root / cob_date
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._mkts: int = mkts

        yahoo_runs: dict[str, dict[str, str | int]] = {
            k: v for k, v in batch_config.items() if v['source'] == 'yahoo'
        }
        norgate_runs: dict[str, dict[str, str | int]] = {
            k: v for k, v in batch_config.items() if v['source'] == 'norgate'
        }

        # Execute all Yahoo runs sequentially
        if yahoo_runs:
            self._run_batch(runs=yahoo_runs, source='yahoo')

        # Execute all Norgate runs sequentially once Yahoo runs are complete
        if norgate_runs:
            self._run_batch(runs=norgate_runs, source='norgate')

    def _run_batch(
        self,
        runs: dict[str, dict[str, str | int]],
        source: str,
    ) -> None:
        """
        Fetch price history for the given source and execute each run
        sequentially.

        Parameters
        ----------
        runs : dict[str, dict[str, str | int]]
            Runs for a single source from batch_config.
        source : str
            Data source; ``'yahoo'`` or ``'norgate'``.
        """
        max_lookback: int = int(max(v['days'] for v in runs.values()))
        logger.info(
            "%s batch: fetching data at max lookback %d days.",
            source.capitalize(),
            max_lookback,
        )

        fetch = _DataFetch.yahoo if source == 'yahoo' else _DataFetch.norgate
        source_metadata, full_price_history, source_mappings = fetch(
            mkts=self._mkts, max_lookback=max_lookback
        )

        for label, run_config in runs.items():
            days: int = int(run_config['days'])
            logger.info("%s run '%s': days=%d.", source.capitalize(), label, days)
            try:
                _execute_run(
                    label=label,
                    days=days,
                    source=source,
                    mkts=self._mkts,
                    source_metadata=source_metadata,
                    full_price_history=full_price_history,
                    source_mappings=source_mappings,
                    output_dir=self._output_dir,
                )
            except Exception:
                logger.exception("%s run '%s' failed.", source.capitalize(), label)


# ---------------------------------------------------------------------------
# Parallel batch
# ---------------------------------------------------------------------------

class TrendBatchParallel:
    """
    Execute the trend strength pipeline for a labelled set of runs using
    parallel worker processes.

    Parameters
    ----------
    batch_config : dict[str, dict[str, str | int]]
        Mapping of output label to run parameters. Each entry must
        contain ``'days'`` (int) and ``'source'`` (str: ``'yahoo'`` or
        ``'norgate'``).
    cob_date : str
        Close-of-business date in YYYYMMDD format; used as the output
        subfolder name.
    output_root : Path
        Root directory under which the cob_date subfolder is created.
    mkts : int
        Number of markets for bar and market chart panels, applied
        uniformly across all runs.
    max_workers : int, optional
        Maximum number of worker processes per source phase. The default
        is 5.
    """

    def __init__(
        self,
        batch_config: dict[str, dict[str, str | int]],
        cob_date: str,
        output_root: Path,
        mkts: int,
        max_workers: int = 5,
    ) -> None:
        self._output_dir: Path = output_root / cob_date
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._mkts: int = mkts
        self._max_workers: int = max_workers

        yahoo_runs: dict[str, dict[str, str | int]] = {
            k: v for k, v in batch_config.items() if v['source'] == 'yahoo'
        }
        norgate_runs: dict[str, dict[str, str | int]] = {
            k: v for k, v in batch_config.items() if v['source'] == 'norgate'
        }

        # Execute all Yahoo runs in parallel
        if yahoo_runs:
            self._run_parallel_batch(runs=yahoo_runs, source='yahoo')

        # Execute all Norgate runs in parallel once Yahoo runs are complete
        if norgate_runs:
            self._run_parallel_batch(runs=norgate_runs, source='norgate')

    def _run_parallel_batch(
        self,
        runs: dict[str, dict[str, str | int]],
        source: str,
    ) -> None:
        """
        Fetch price history for the given source then submit all runs to
        a ``ProcessPoolExecutor``.

        Parameters
        ----------
        runs : dict[str, dict[str, str | int]]
            Runs for a single source from batch_config.
        source : str
            Data source; ``'yahoo'`` or ``'norgate'``.
        """
        max_lookback: int = int(max(v['days'] for v in runs.values()))
        logger.info(
            "%s parallel batch: fetching data at max lookback %d days.",
            source.capitalize(),
            max_lookback,
        )

        fetch = _DataFetch.yahoo if source == 'yahoo' else _DataFetch.norgate
        source_metadata, full_price_history, source_mappings = fetch(
            mkts=self._mkts, max_lookback=max_lookback
        )

        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    _execute_run,
                    label=label,
                    days=int(run_config['days']),
                    source=source,
                    mkts=self._mkts,
                    source_metadata=source_metadata,
                    full_price_history=full_price_history,
                    source_mappings=source_mappings,
                    output_dir=self._output_dir,
                ): label
                for label, run_config in runs.items()
            }

            for future in as_completed(futures):
                label = futures[future]
                try:
                    completed_label: str = future.result()
                    logger.info(
                        "%s run '%s' completed.", source.capitalize(), completed_label
                    )
                except Exception:
                    logger.exception(
                        "%s run '%s' failed.", source.capitalize(), label
                    )
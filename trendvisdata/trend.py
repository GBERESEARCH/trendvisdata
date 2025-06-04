"""
Calculate Trend Strength across various markets / asset classes and graph
results

"""
import copy
import norgatedata
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import BDay, DateOffset
from trendvisdata.chart_data import Data
from trendvisdata.sector_mappings import sectmap
from trendvisdata.trend_params import trend_params_dict
from trendvisdata.trend_data import Fields, TrendRank
from trendvisdata.market_data import NorgateExtract, YahooExtract, MktUtils
from typing import cast


class TrendStrength():
    """
    Create Trend Strength data and display results

    Parameters
    ----------
    chart_dimensions : Tuple
        Tuple of height, width for market chart.
    chart_mkts : Int
        Number of markets for market chart.
    days : Int
        The number of days price history.
    end_date : Str
        End Date represented as a string in the
        format 'YYYY-MM-DD'.
    indicator_type : Str
        The indicator to plot. Choose from 'adx', 'ma_cross',
        'price_cross', 'rsi', 'breakout'.
    lookback : Int
        Number of days history if dates are not specified
    mkts : Int
        Number of markets for barchart or linegraph.
    norm : Bool
        Whether the prices have been normalised.
    pie_tenor : Int / Tuple
        The time period of the indicator. For the Moving Average
        crossover this is a tuple from the following pairs: (5, 200),
        (10, 30), (10, 50), (20, 50), (30, 100), (50, 200). For the
        other indicators this is an integer from the list: 10, 20, 30,
        50, 100, 200.
    sector_level : Int
        The level of granularity of the assets.
        For Commodities the choices are:
            1:'Asset Class',
            2:'Broad Sector',
            3:'Mid Sector',
            4:'Narrow Sector',
            5:'Underlying'.
        For Equities the choices are:
            1:'Sector',
            2:'Industry Group',
            3:'Industry',
            4:'Sub-Industry',
            5:'Security'
    source : Str
        The source of the market data. 'norgate' or 'yahoo'. The default is
        'norgate'.
    start_date : Str
        Start Date represented as a string in the
        format 'YYYY-MM-DD'.
    tickers : List
        List of tickers, represented as strings.
    ticker_limit : Int
        Flag to select only the first n markets. The default
        is None.
    trend : Str
        Flag to select most or least trending markets.
        Select from: 'up' - strongly trending upwards,
                     'down - strongly trending downwards,
                     'neutral' - weak trend,
                     'strong' - up and down trends,
                     'all' - up down and weak trends
        The default is 'strong' which displays both up-trending
        and down-trending markets.

    Returns
    -------
    None.

    """
    def __init__(self, **kwargs) -> None:

        # Import dictionary of default parameters
        self.default_dict = copy.deepcopy(trend_params_dict)

        # Import dictionary of sector mappings
        mappings = copy.deepcopy(sectmap)

        # Store initial inputs
        inputs = {}
        for key, value in kwargs.items():
            inputs[key] = value

        # Initialise system parameters
        params = self._init_params(inputs)

        # Import the data from Norgate Data
        if params['source'] == 'norgate':
            params, tables, mappings = self.prep_norgate(
                 params=params, mappings=mappings)

        # Or from Yahoo Finance
        elif params['source'] == 'yahoo':
            params, tables, mappings = self.prep_yahoo(
                params=params, mappings=mappings)

        # Calculate the technical indicator fields and Trend Strength table
        tables = self.trend_calc(
            params=params, tables=tables, mappings=mappings)

        # Generate list of top trending securities
        top_trends, tables = self.top_trend_tickers(
            params=params, tables=tables)

        # Generate data dictionary for graphing via API
        data_dict = Data.get_all_data(params=params, tables=tables)

        self.top_trends = top_trends
        self.tables = tables
        self.params = params
        self.mappings = mappings
        self.data_dict = data_dict


    @staticmethod
    def _init_params(inputs: dict) -> dict:
        """
        Initialise parameter dictionary
        Parameters
        ----------
        inputs : Dict
            Dictionary of parameters supplied to the function.

        Returns
        -------
        params : Dict
            Dictionary of parameters.
        """
        # Copy the default parameters
        params = copy.deepcopy(trend_params_dict['df_params'])

        # For all the supplied arguments
        for key, value in inputs.items():

            # Replace the default parameter with that provided
            params[key] = value

        return params


    @staticmethod
    def prep_norgate(
        params: dict,
        mappings: dict) -> tuple[dict, dict, dict]:
        """
        Create dataframes of prices, extracting data from Norgate Data.

        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.
        mappings : Dict
            Dictionary of sector mappings.

        Returns
        -------
        params : Dict
            Dictionary of key parameters.
        tables : Dict
            Dictionary of key tables.
        mappings : Dict
            Dictionary of sector mappings.

        """

        # Set the asset type to 'CTA'
        params['asset_type'] = 'CTA'

        # If a list of tickers are not supplied, run the function to collect
        # available tickers
        if params['tickers'] is None:
            params = NorgateExtract.get_norgate_tickers(params=params)

        # Set the start and end dates
        params = MktUtils.date_set(params)

        # Dictionary to store data tables
        tables = {}

        # Create dictionaries of DataFrames of prices and ticker names
        params, tables, mappings = NorgateExtract.import_norgate(
            params=params, tables=tables, mappings=mappings)

        # Remove tickers with short history
        tables = MktUtils.ticker_clean(params=params, tables=tables)

        return params, tables, mappings


    @staticmethod
    def prep_yahoo(
        params: dict,
        mappings: dict) -> tuple[dict, dict, dict]:
        """
        Create dataframes of prices, extracting data from Yahoo Finance.

        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.
        mappings : Dict
            Dictionary of sector mappings.

        Returns
        -------
        params : Dict
            Dictionary of key parameters.
        tables : Dict
            Dictionary of key tables.
        mappings : Dict
            Dictionary of sector mappings.

        """

        # Create list of tickers, dictionary of ticker names from
        # Wikipedia
        params, mappings = YahooExtract.ticker_extract(
            params=params, mappings=mappings)

        # Set short_name_dict = name_dict
        params['ticker_short_name_dict'] = params['ticker_name_dict']

        # Set the asset type to 'Equity'
        params['asset_type'] = 'Equity'

        # Set the start and end dates
        params = MktUtils.date_set(params)

        # Dictionary to store data tables
        tables = {}

        # Create dictionaries of DataFrames of prices and ticker names
        params, tables = YahooExtract.import_yahoo(params, tables)

        # Remove tickers with short history
        tables = MktUtils.ticker_clean(params=params, tables=tables)

        return params, tables, mappings


    @staticmethod
    def trend_calc(
        params: dict,
        tables: dict,
        mappings: dict) -> dict:
        """
        Calculate the technical indicator fields and Trend Strength table

        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.
        tables : Dict
            Dictionary of key tables.
        mappings : Dict
            Dictionary of sector mappings.

        Returns
        -------
        tables : Dict
            Dictionary of key tables.

        """
        # Calculate the technical indicator fields
        tables['ticker_dict'] = Fields.generate_fields(
            params, tables['raw_ticker_dict'])

        # Calculate the Trend Strength table
        tables['barometer'] = Fields.generate_trend_strength(
            params=params, ticker_dict=tables['ticker_dict'],
            sector_mappings_df=mappings['sector_mappings_df'])

        return tables


    @staticmethod
    def top_trend_tickers(
        params: dict,
        tables: dict) -> tuple[dict, dict]:
        """
        Prepare list of top trending securities.

        Parameters
        ----------
        params : Dict
            Dictionary of key parameters.
        tables : Dict
            Dictionary of key tables.

        Returns
        -------
        ticker_list : List
                List of top trending securities.
        tables : Dict
            Dictionary of key tables.

        """
        # Generate list of top trending securities
        top_trends, tables = TrendRank.top_trend_calc(
            tables, params)

        return top_trends, tables


class ReturnsHistory():
    """
    Generate dictionary of lists of various return periods.
    """
    def __init__(self, start_date, end_date) -> None:
        self.tenor_mappings = trend_params_dict['df_params']['tenor_mappings']
        self.returns = self.generate_returns(
            start_date,
            end_date,
            self.tenor_mappings
            )


    @staticmethod
    def get_tickers() -> list:
        """
        Get all the tickers from the Norgate Futures package (other than individual
        contracts)

        Returns
        -------
        lim_tickers : List
            List of ticker codes.

        """
        all_tickers = []
        alldatabasenames = norgatedata.databases()
        databasenames = alldatabasenames[:4]
        databasenames.append(alldatabasenames[-1])

        for item in databasenames:
            tickers = norgatedata.database_symbols(item)
            for ticker in tickers: # type: ignore
                all_tickers.append(ticker)

        lim_tickers = []
        for ticker in all_tickers:
            if ticker[-4:] != '_CCB':
                lim_tickers.append(ticker)

        return lim_tickers


    @staticmethod
    def get_history(
            start_date: str,
            end_date: str,
            tickers: list) -> pd.DataFrame:
        """
        Create DataFrame of closing price histories for provided list of tickers

        Parameters
        ----------
        start_date : String
            The start date for comparison. The format is YYYY-MM-DD
        end_date : String
            The end date for comparison. The format is YYYY-MM-DD
        tickers : List
            List of ticker codes.

        Returns
        -------
        history : DataFrame
            Pandas DataFrame of history of closing prices for each ticker in
            tickers.

        """
        history = pd.DataFrame()

        # If end date is not supplied, set to previous working day
        if end_date is None:
            end_date_as_dt = (dt.datetime.today() - BDay(1)).date()
            end_date = str(end_date_as_dt)

        # If start date is not supplied, set to today minus lookback period
        if start_date is None:
            start_date_as_dt = (
                dt.datetime.today()
                - pd.Timedelta(days=trend_params_dict['df_params']['lookback']*(365/250))).date()
            start_date = str(start_date_as_dt)

        for ticker in tickers:
            data = norgatedata.price_timeseries(
                ticker, start_date=start_date,
                end_date=end_date,
                format='pandas-dataframe',)

            ticker_name = norgatedata.security_name(ticker)

            if len(history) == 0:
                history[ticker_name] = data['Close'] # type: ignore
            else:
                data.rename(columns={'Close': ticker_name}, inplace=True) # type: ignore
                history = pd.concat((history, data[ticker_name]), axis=1) # type: ignore

        history.ffill(inplace=True)

        history.dropna(inplace=True)

        return history

    
    @classmethod
    def get_prices(
        cls, 
        history: pd.DataFrame, 
        tenor_mappings: dict, 
        short_label: bool = True) -> dict:
        """
        Calculate prices for each ticker and store these in a dictionary of lists

        Parameters
        ----------
        history : DataFrame
            Pandas DataFrame of history of closing prices for each ticker in
            tickers.
        tenor_mappings : Dict
            Dictionary of mappings from week / month to day and day to column
            headings.

        Returns
        -------
        returns_dict : Dict
            Dictionary of lists of various return periods.

        """
        tenor_dates = cls._get_tenor_dates(history, tenor_mappings, short_label)
        
        prices_df = pd.DataFrame()
        prices_df.index = history.columns
        
        for column in history.columns:
            for tenor_label, date_val in tenor_dates.items():
                prices_df.loc[column, tenor_label] = history.loc[date_val, column]
        
        data = prices_df.T.to_dict(orient='list')
        tenors = list(prices_df.columns)
        prices_array = []
        for key, value in data.items():
            asset_dict = {'label': key}
            for num, tenor in enumerate(tenors):
                asset_dict[tenor] = value[num]
            prices_array.append(asset_dict)
        
        return {
            'data': prices_array,
            'labels': tenors
        }
    

    @classmethod
    def get_returns(
        cls, 
        history: pd.DataFrame, 
        tenor_mappings: dict, 
        short_label: bool = True) -> dict:
        """
        Calculate returns for each ticker and store these in a dictionary of lists

        Parameters
        ----------
        history : DataFrame
            Pandas DataFrame of history of closing prices for each ticker in
            tickers.
        tenor_mappings : Dict
            Dictionary of mappings from week / month to day and day to column
            headings.

        Returns
        -------
        returns_dict : Dict
            Dictionary of lists of various return periods.

        """
        tenor_dates = cls._get_tenor_dates(
            history=history, 
            tenor_mappings=tenor_mappings, 
            short_label=short_label
            )
        
        returns_df = pd.DataFrame()
        returns_df.index = history.columns
        end_date = history.index[-1]
        
        for column in history.columns:
            current_price = cast(float, history.loc[end_date, column])
            
            for tenor_label, date_val in tenor_dates.items():
                past_price = cast(float, history.loc[date_val, column])
                returns_df.loc[column, tenor_label] = (
                    (current_price - past_price) / past_price * 100
                )
        
        data = returns_df.T.to_dict(orient='list')
        tenors = list(returns_df.columns)
        returns_array = []
        for key, value in data.items():
            asset_dict = {'label': key}
            for num, tenor in enumerate(tenors):
                asset_dict[tenor] = value[num]
            returns_array.append(asset_dict)
        
        return {
            'data': returns_array,
            'labels': tenors
        }


    @staticmethod
    def _get_tenor_dates(
        history: pd.DataFrame, 
        tenor_mappings: dict, 
        short_label: bool) -> dict:
        """
        Calculate the actual dates to use for each tenor period.
        
        Returns
        -------
        tenor_dates : dict
            Dictionary mapping tenor labels to actual dates
        """
        def find_nearest_business_day(target_date, available_dates):
            """Find the nearest available business day to the target date."""
            if target_date in available_dates:
                return target_date
            
            available_dates_sorted = sorted(available_dates)
            
            # Find dates on or after target
            future_dates = [d for d in available_dates_sorted if d >= target_date]
            if future_dates:
                return future_dates[0]
            
            # If target is before our data starts, take the earliest available
            return available_dates_sorted[0]
        
        end_date = history.index[-1]
        available_dates = set(history.index)
        
        days = tenor_mappings['days']
        weeks = tenor_mappings['weeks']
        months = tenor_mappings['months']
        if short_label:
            labels = tenor_mappings['short_labels']
        else:                
            labels = tenor_mappings['labels']
        
        tenor_dates = {}
        
        # Days: convert trading day offset to actual date
        for day in days:
            actual_date = history.index[-day-1]
            tenor_dates[labels[day]] = actual_date
        
        # Weeks: use calendar arithmetic with business day adjustment
        for week, week_day in weeks.items():
            target_date = end_date + DateOffset(weeks=-week)
            actual_date = find_nearest_business_day(target_date, available_dates)
            tenor_dates[labels[week_day]] = actual_date
        
        # Months: use calendar arithmetic with business day adjustment
        for month, month_day in months.items():
            target_date = end_date + DateOffset(months=-month)
            actual_date = find_nearest_business_day(target_date, available_dates)
            tenor_dates[labels[month_day]] = actual_date
        
        return tenor_dates


    @classmethod
    def generate_returns(
            cls,
            start_date: str,
            end_date: str,
            tenor_mappings: dict) -> dict:
        """
        Generate dictionary of lists of various return periods.

        Parameters
        ----------
        start_date : String
            The start date for comparison. The format is YYYY-MM-DD
        end_date : String
            The end date for comparison. The format is YYYY-MM-DD
        tenor_mappings : Dict
            Dictionary of mappings from week / month to day and day to column
            headings.

        Returns
        -------
        returns : Dict
            Dictionary of lists of various return periods.

        """
        tickers = cls.get_tickers()
        history = cls.get_history(start_date, end_date, tickers)
        returns = cls.get_returns(
            history=history, 
            tenor_mappings=tenor_mappings, 
            short_label=True
            )
        returns_long = cls.get_returns(
            history=history, 
            tenor_mappings=tenor_mappings, 
            short_label=False
            )
        prices = cls.get_prices(
            history=history, 
            tenor_mappings=tenor_mappings, 
            short_label=True
            )
        prices_long = cls.get_prices(
            history=history, 
            tenor_mappings=tenor_mappings, 
            short_label=False
            )

        return {
            'returns': returns,
            'returns_long': returns_long,
            'prices': prices,
            'prices_long': prices_long
        }

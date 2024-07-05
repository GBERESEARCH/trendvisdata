import numpy as np
import pandas as pd
import collections
from trendvisdata.chart_prep import Formatting

class Data():
    """
    Create data dictionaries to display various charts of Trend Strength

    """
    @classmethod
    def get_all_data(cls, params, tables):
        """
        Create data dictionary for graphing of barchart, returns and market charts.

        Parameters
        ----------
        params : Dict
            mkts : Int
                Number of markets to chart. The default is 20.
            chart_mkts : Int (Optional)
                Number of markets to chart. The default is None.
            chart_dimensions : Tuple
                Width and height to determine the number of markets to chart. The
                default is (8, 5)
            trend : Str
                Flag to select most or least trending markets.
                Select from: 'up' - strongly trending upwards,
                            'down - strongly trending downwards,
                            'neutral' - weak trend,
                            'strong' - up and down trends,
                            'all' - up down and weak trends
                The default is 'strong' which displays both up-trending
                and down-trending markets.
            days : Int
                Number of days of history. The default is 60.
        tables : Dict
            Dictionary of key tables.

        Returns
        -------
        data_dict : Dict
            Data dictionary for graphing of barchart, returns and market charts.

        """
        data_dict = {}
        barometer = tables['barometer']
        data_dict['bar_dict'] = cls.get_bar_data(barometer=barometer, params=params)
        data_dict['returns_dict'] = cls.get_returns_data(
            params=params, tables=tables)
        data_dict['market_dict'] = cls.get_market_chart_data(
            params=params, tables=tables)
        
        return data_dict


    @classmethod
    def get_bar_data(cls, barometer: pd.DataFrame, params: dict) -> dict:
        """
        Create data dictionary for plotting barchart trends.

        Parameters
        ----------
        barometer : DataFrame
            DataFrame showing trend strength for each ticker.
        mkts : Int
            Number of markets to chart. The default is 20.

        Returns
        -------
        bar_dict : Dict
            Data dictionary for plotting barchart trends.

        """
        # Create default dict to add keys not found
        bar_dict = collections.defaultdict(dict)
        mkts = params['mkts']

        # Create entries for up trend
        bar_dict = cls._barometer_up(barometer, mkts, bar_dict)
        
        # Create entries for down trend
        bar_dict = cls._barometer_down(barometer, mkts, bar_dict)

        # Create entries for neutral trend
        bar_dict = cls._barometer_neutral(barometer, mkts, bar_dict)

        # Create entries for strong trend
        bar_dict = cls._barometer_strong(barometer, mkts, bar_dict)

        # Convert back to regular dict
        bar_dict = dict(bar_dict)

        return bar_dict
    
    
    @classmethod
    def _barometer_up(cls, barometer, mkts, bar_dict):
        # Create entries for up trend
        barometer_up = barometer.sort_values(
            by=['Trend Strength %'], ascending=True)
        short_name = list(barometer_up['Short_name'].iloc[-mkts:])
        trend_strength = np.round(
            np.array(barometer_up['Trend Strength %'].iloc[-mkts:]), 4)
        trend_color = list(barometer_up['Trend Color'].iloc[-mkts:])
        bar_dict = cls._bar_arrays(
            bar_dict, short_name, trend_strength, trend_color, direction='up')
        
        return bar_dict
    

    @classmethod
    def _barometer_down(cls, barometer, mkts, bar_dict):
        # Create entries for down trend
        barometer_down = barometer.sort_values(
            by=['Trend Strength %'], ascending=False)
        short_name = list(barometer_down['Short_name'].iloc[-mkts:])
        trend_strength = np.round(
            np.array(barometer_down['Trend Strength %'].iloc[-mkts:]), 4)
        trend_color = list(
            barometer_down['Trend Color'].iloc[-mkts:])
        bar_dict = cls._bar_arrays(
            bar_dict, short_name, trend_strength, trend_color, direction='down')
        
        return bar_dict

        
    @classmethod
    def _barometer_neutral(cls, barometer, mkts, bar_dict):
        # Create entries for neutral trend
        barometer_neutral = barometer.sort_values(
            by=['Absolute Trend Strength %'], ascending=True)
        short_name = list(barometer_neutral['Short_name'].iloc[:mkts])
        trend_strength = np.round(
            np.array(barometer_neutral['Trend Strength %'].iloc[:mkts]), 4)
        trend_color = list(barometer_neutral['Trend Color'].iloc[:mkts])
        bar_dict = cls._bar_arrays(
            bar_dict, short_name, trend_strength, trend_color, direction='neutral')
        
        return bar_dict


    @classmethod
    def _barometer_strong(cls, barometer, mkts, bar_dict):
        # Create entries for strong trend
        barometer_neutral = barometer.sort_values(
            by=['Absolute Trend Strength %'], ascending=True)
        short_name = list(
            barometer_neutral['Short_name'].iloc[-mkts:])
        trend_strength = np.round(
            np.array(barometer_neutral['Trend Strength %'].iloc[-mkts:]), 4)
        trend_color = list(
            barometer_neutral['Trend Color'].iloc[-mkts:])
        bar_dict = cls._bar_arrays(
            bar_dict, short_name, trend_strength, trend_color, direction='strongly')

        return bar_dict


    @classmethod
    def _bar_arrays(
        cls,
        bar_dict, 
        short_name, 
        trend_strength, 
        trend_color, 
        direction
        ):
        bar_dict[direction] = collections.defaultdict(dict)
        bar_dict[direction]['arrays']['short_name'] = short_name
        bar_dict[direction]['arrays']['trend_strength'] = trend_strength
        bar_dict[direction]['arrays']['trend_color'] = trend_color
        bar_dict[direction]['json'] = cls._json_array(
            short_name, trend_strength, trend_color)
        bar_dict[direction]['titlestr'] = direction.capitalize()

        bar_dict[direction] = dict(bar_dict[direction])

        return bar_dict


    @staticmethod
    def _json_array(short_name, trend_strength, trend_color):
        bar_trim = pd.DataFrame()
        bar_trim['Short Name'] = short_name
        bar_trim['Trend Strength'] = trend_strength
        bar_trim['Trend Color'] = trend_color

        json_dict = bar_trim.to_dict(orient='index')

        json_array = []
        for key in json_dict.keys():
            json_array.append(json_dict[key])

        return json_array
    

    @classmethod
    def get_returns_data(cls, params: dict, tables: dict) -> dict:        
        """
        Create data dictionary for plotting line graph trends.

        Parameters
        ----------
        params : Dict
            mkts : Int
                Number of markets to chart. The default is 5.
            trend : Str
                Flag to select most or least trending markets.
                Select from: 'up' - strongly trending upwards,
                            'down - strongly trending downwards,
                            'neutral' - weak trend,
                            'strong' - up and down trends,
                            'all' - up down and weak trends
                The default is 'strong' which displays both up-trending
                and down-trending markets.
            days : Int
                Number of days of history. The default is 60.
        tables : Dict
            Dictionary of key tables.

        Returns
        -------
        returns_dict : Dict
            Dictionary of data to create a line graph of normalised price history.

        """
        # Generate DataFrame of normalized returns
        tenor = Formatting.create_normalized_data(params=params, tables=tables)
        #tenor.index = tenor.index.astype(pd.DatetimeIndex)
        tenor.index = tenor.index.date.astype(str) # type: ignore comment;

        # Create empty returns dict & add returns and labels
        returns_dict = {}
        returns_dict['time_series'] = {}
        for num, label in enumerate(tenor.columns):
            returns_dict['time_series'][num] = {}
            returns_dict['time_series'][num]['label'] = label
            returns_dict['time_series'][num]['data'] = tenor[label].to_dict()

        returns_dict['time_series'] = cls._round_floats(
            returns_dict['time_series']
            )
        returns_dict['xlabel'] = 'Date'
        returns_dict['ylabel'] = 'Return %'
        returns_dict['line_labels'] = tenor.columns.to_list()
        returns_dict['chart_title'] = (
            'Relative Return Over Last ' +
            str(len(tenor)) +
            ' Trading Days' +
            ' - ' +
            params['end_date']
            )

        return returns_dict
    

    @classmethod
    def get_market_chart_data(cls, params: dict, tables: dict) -> dict:
        """
        Create a data dictionary for plotting a summary of the strength of trend 
        across markets.

        Parameters
        ----------
        params : Dict
            chart_mkts : Int (Optional)
                Number of markets to chart. The default is None.
            chart_dimensions : Tuple
                Width and height to determine the number of markets to chart. The 
                default is (8, 5)
            trend : Str
                Flag to select most or least trending markets.
                Select from: 'up' - strongly trending upwards,
                            'down - strongly trending downwards,
                            'neutral' - weak trend,
                            'strong' - up and down trends,
                            'all' - up down and weak trends
                The default is 'strong' which displays both up-trending
                and down-trending markets.
            days : Int
                Number of days of history. The default is 60.
        tables : Dict
            Dictionary of key tables.

        Returns
        -------
        market_dict : Dict
            Data dictionary for plotting a summary of the strength of trend across 
            markets

        """
        if params['chart_mkts'] is not None:
            params = Formatting.create_mkt_dims(params)
        
        params['num_charts'] = int(
            params['chart_dimensions'][0] * params['chart_dimensions'][1])
        
        data_list = Formatting.create_data_list(
            params=params, barometer=tables['barometer'], market_chart=True,
            num_charts=params['num_charts'])

        market_dict = collections.defaultdict(dict)
        market_dict['tickers'] = collections.defaultdict(dict)
                
        for num, ticker in enumerate(data_list):
            market_dict['tickers'][num]['label'] = params[
                'ticker_short_name_dict'][ticker]
            market_dict['tickers'][num]['ticker'] = ticker

            market_dict['tickers'][num]['axis_dates'] = (
                tables['ticker_dict'][ticker].index[-params['days']:]
                ).date.tolist()
            market_dict['tickers'][num][
                'axis_prices_norm'] = cls._round_floats(
                    np.array(tables['ticker_dict'][ticker]['Close'][
                        -params['days']:].div(tables['ticker_dict'][
                            ticker]['Close'][-params['days']:].iloc[0])
                            .mul(100)).tolist()
                            )

            market_dict['tickers'][num]['axis_prices'] = cls._round_floats(
                np.array(tables['ticker_dict'][ticker]['Close'][
                    -params['days']:]).tolist())
        
        market_dict = dict(market_dict)
        market_dict['tickers'] = dict(market_dict['tickers'])
        market_dict['chart_title'] = Formatting.get_chart_title(params=params) # type: ignore comment;
        
        return market_dict
    

    @classmethod
    def _round_floats(cls, obj):
        if isinstance(obj, float): return round(obj, 2)
        if isinstance(obj, dict): return {
            k: cls._round_floats(v) for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)): return [
            cls._round_floats(x) for x in obj
            ]
        
        return obj
    
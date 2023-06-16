def SMA(df, base, target, period):
    """
    Function to compute Simple Moving Average (SMA)

    Args :
        df : Pandas DataFrame
        base : String indicating the column name from which the SMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles

    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """
    df[target] = df[base].rolling(window=period).mean()
    return df

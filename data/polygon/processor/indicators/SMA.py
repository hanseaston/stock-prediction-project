def SMA(df, base, target, period):
    """
    Function to compute Simple Moving Average (SMA)

    Args :
        df : pandas dataframe
        base : the column name from which the ratio needs to be computed
        target : the column name to which the computed data needs to be stored
        period : number of trading timeframe to calculate the SMA over

    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """
    df[target] = df[base].rolling(window=period).mean()
    return df

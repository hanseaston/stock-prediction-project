def LagRatio(df, base, target, period):
    """
    Function to compute the ratio between day X and day (X - period)

    Args :
        df : pandas dataframe
        base: the column name from which the ratio needs to be computed
        period: number of trading timeframe to calculate the ratio

    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """
    df[target] = df[base] / df[base].shift(period) - 1
    return df

def LagRatio(df, base, target, period):
    df[target] = df[base] / df[base].shift(period) - 1
    return df

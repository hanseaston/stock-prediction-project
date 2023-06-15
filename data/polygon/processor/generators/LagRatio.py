def LagRatio(df, base, target, period):
    if target not in df.columns:
        df[target] = df[base] / df[base].shift(period) - 1
    return df

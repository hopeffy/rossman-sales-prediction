import pandas as pd

def _create_date_features(df):
    """Creates time-based features from the 'Date' column."""
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    return df

def _create_competition_features(df):
    """Creates features related to competitors."""
    df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    df['CompetitionOpen'] = (df['Year'] - df['CompetitionOpenSinceYear']) * 12 + \
                            (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: max(x, 0))
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    return df

def _is_promo2_active(row):
    """Helper function to check if Promo2 is active for a given row."""
    if row['Promo2'] == 0:
        return 0
    
    promo2_start_year = int(row['Promo2SinceYear'])
    promo2_start_week = int(row['Promo2SinceWeek'])
    current_year = row['Year']
    current_week = row['WeekOfYear']

    if current_year < promo2_start_year:
        return 0
    if current_year == promo2_start_year and current_week < promo2_start_week:
        return 0

    month_str = row['Date'].strftime('%b')
    if month_str in row['PromoInterval']:
        return 1
    else:
        return 0

def _create_promo2_features(df):
    """Creates features related to the ongoing Promo2."""
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna('', inplace=True)
    df['IsPromo2'] = df.apply(_is_promo2_active, axis=1)
    return df

def engineer_features(df):
    """
    Main function to engineer all features for the Rossmann sales model.

    Args:
        df (pd.DataFrame): The input dataframe (merged train or test data).

    Returns:
        pd.DataFrame: The dataframe with engineered features.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = _create_date_features(df)
    df = _create_competition_features(df)
    df = _create_promo2_features(df)
    
    print("Feature engineering complete.")
    return df


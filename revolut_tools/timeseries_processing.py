import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_daily_transactions_df(users, transactions, analysis_end_date):
    """
    Given a transactions dataframe and a users DataFrame
     generate a view table of the daily transactions that each user made on each day.

    Args:
        users: users dataframe
        transactions: transactions dataframe
        analysis_end_date: date that we want to limit the analysis at

    Returns:
        daily_transactions: table with the daily transactions that each user made each day
    """

    # Filter data by time
    analysis_end_date = pd.to_datetime(analysis_end_date)
    transactions = transactions[
    transactions['created_date']<=analysis_end_date
    ].copy()
    users = users[users['created_date']<=analysis_end_date].copy()

    # Define censored point
    users['censored_at'] = (analysis_end_date - users['created_date']).dt.days

    #Create a dataframe with all days a user was alive
    daily_target = pd.DataFrame(
        users['user_id'].repeat(users['censored_at']).reset_index(drop = True)
    )
    daily_target['days_alive'] = daily_target.groupby(['user_id']).cumcount()+1

    daily_target = daily_target.merge(users[['user_id', 'created_date']])
    daily_target['date'] = (
        daily_target['created_date'] +
        pd.to_timedelta(daily_target['days_alive'], unit='d')
    ).dt.date

    daily_target = daily_target.drop('created_date', axis=1)


    # Calculate daily transactions per user
    transactions['order_week'] = transactions['created_date'].dt.date
    users['user_created_date'] = users['created_date']
    transactions_cohort = transactions.merge(
    users[['user_id', 'user_created_date']], on='user_id'
    )
    transactions_cohort['days_alive'] = (
        transactions_cohort['created_date']
        -
        transactions_cohort['user_created_date']
    ).dt.days
    daily_transactions = transactions_cohort.groupby(
        ['user_id', 'days_alive']
    )['transaction_id'].count().reset_index()

    daily_transactions.rename(
    {'transaction_id':'transaction_number'}, axis=1, inplace=True
    )


    # Merge daily transactions per user with daily_target table
    daily_transactions = daily_target.merge(
    daily_transactions, on=['user_id', 'days_alive'], how='left'
    )
    daily_transactions['transaction_number'] = daily_transactions[
    'transaction_number'
    ].fillna(0)
    return daily_transactions


def generate_notication_actions_df (notifications, transactions, n_days=1):
    """
    Given a transactions dataframe and a notifications DataFrame
     generate a view of the number of transactions that happend after n_days
     after a notification.

    Args:
        users: users dataframe
        transactions: transactions dataframe
        n_days: days we want to check the transactions after the notification

    Returns:
        notification_actions_grouped: table with the transactions that happend after n_days
        after each notification
    """
    # This would be much more efficient and simple to do on SQL
    notification_actions = notifications.merge(
        transactions,
        on='user_id',
        suffixes=['_notification', '_transaction']
    )

    # Delete transactions that happened before notification in order to make process faster
    notification_actions = notification_actions[
        notification_actions['created_date_notification'] < notification_actions['created_date_transaction']
    ]


    notification_actions['notification_transaction_timelapse'] = (
        notification_actions['created_date_transaction']
        -
        notification_actions['created_date_notification']
    ).dt.days

    #Calculate the number of transactions on the given time interval
    notification_actions_grouped = notification_actions.groupby(
        ['user_id', 'created_date_notification', 'reason', 'channel']
    )['notification_transaction_timelapse'].apply(
        lambda x: ((x>=0) & (x<=n_days)).sum()
    ).reset_index(name='count')
    notification_actions_grouped.rename({'count':'action_count'}, axis=1, inplace=True)

    notification_actions_grouped['engaged'] = notification_actions_grouped['action_count']>0

    return notification_actions_grouped

def add_user_cohort_info(df, users, time_column_name='created_date'):
    """
    Add the cohort information (cohort date and days alive) to a df

    Args:
        df: dataframe with timeseries data of users
        users: users dataframe
        time_column_name: name of the time dataframe on df

    Returns:
        df_cohort: df with the cohort information
    """
    df['cohort_date'] = df[time_column_name].dt.date
    users['user_created_date'] = users['created_date']

    df_cohort = df.merge(users[['user_id', 'user_created_date']], on='user_id')
    df_cohort['days_alive'] = (
        df_cohort[time_column_name] - df_cohort['user_created_date']
    ).dt.days
    df_cohort.drop(['user_created_date', 'cohort_date'], axis=1, inplace=True)

    return df_cohort

def add_engagement_to_daily_transactions(daily_transactions, engagement_period):
    label = 'last_{}_days_transactions'.format(engagement_period)
    #Calculate engament level with based on the window of engagement_period
    daily_transactions[label] = (
        daily_transactions.groupby(
            'user_id'
        )['transaction_number'].shift(1).rolling(engagement_period, min_periods=1).sum()
    ).fillna(0).values
    return daily_transactions

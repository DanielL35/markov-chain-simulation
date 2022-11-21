import pandas as pd
import os
from datetime import timedelta
from datetime import datetime
import random
import numpy as np

def get_df(dir_path):
    # get all csv files in data folder
    file_list = os.listdir(dir_path)
    # read in all customer data for all days in the week
    days = {}
    for file in file_list:
        if file.endswith('.csv'):
            days[file] = pd.read_csv(f'{dir_path}{file}',
                                     sep=';',
                                     parse_dates=[0]
                                     )
    # save all data frames in a dict
    for key in days.keys():
        days[key]['customer_no'] = days[key]['customer_no'].apply(
            lambda x: f'{key[:3]}_{x}')
    # concate all data frames
    df = pd.concat([days['monday.csv'],
                    days['tuesday.csv'],
                    days['wednesday.csv'],
                    days['thursday.csv'],
                    days['friday.csv']]
                  )
    df.reset_index(inplace=True)
    return df

def get_prob(dir_path):
    ''' returns probability matrix for markov chain '''
    df = get_df(dir_path)
    # add all df's together
    timestamp, location, customer_no, location_shift, first_loc = [], [], [], [], []

    for index, customer in enumerate(df['customer_no'].unique()):
        df_customer = df[df['customer_no'] == customer]
        if df_customer.iloc[-1,3] != 'checkout':
            # print(f"customer {customer} has no checkout step")
            continue
        df_customer.reset_index(inplace = True)
        first_loc.append(df_customer['location'][0])
        timestamp.extend(df_customer['timestamp'])
        customer_no.extend(df_customer['customer_no'])
        location.extend(df_customer['location'])
        location_shift.extend(list(df_customer['location'].shift(-1)))

    df_markov = pd.DataFrame({'timestamp': timestamp,
                              'customer_no': customer_no,
                              'location' : location,
                              'location_shift' : location_shift}
                            )
    # generate the probability matrix
    P = pd.crosstab(df_markov['location'],
                    df_markov['location_shift'],
                    normalize=0)
    # check if it makes sense
    test_names = [P.columns[i] for i in range(4)]
    test = [P.iloc[i, :].sum() for i in range(4)]
    print(test_names)
    print(test)
    #  get prob for start location
    prob_init_loc = {i : first_loc.count(i) / len(first_loc) for i in np.unique(
        first_loc)}

    return P, prob_init_loc

def get_poisson_param(dir_path):  # iterate through all customers
    """ Returns lambda for the poisson distribution. """
    df = get_df(dir_path)
    entrance_time, customer_id_li = [], []

    for index, customer in enumerate(df['customer_no'].unique()):
        df_customer = df[df['customer_no'] == customer]
        df_customer.reset_index(inplace=True)
        entrance_time.append(df_customer['timestamp'][0])
        customer_id_li.append(df_customer['customer_no'][0])

    df_entrance_time = pd.DataFrame({
                        'time': entrance_time,
                        'customer': customer_id_li
                                    }
                                    )

    df_entrance_time = df_entrance_time.groupby(by='time').count()

    all_timesteps = []
    for day in [2, 3, 4, 5, 6]:
        base = datetime.strptime(f"0{day}/09/19 07:00", "%d/%m/%y %H:%M")
        num_minutes = (15*60)+1  # open from 7 to 10
        day_list = [base + timedelta(minutes=x) for x in range(num_minutes)]
        all_timesteps.extend(day_list)

    df_all_timesteps = pd.DataFrame({
                        'time': all_timesteps,
                        'new_customers': [0]*len(all_timesteps)
                                    }
                                    )

    df_all_timesteps.set_index('time', inplace=True)
    df_full = pd.concat([df_all_timesteps, df_entrance_time], axis=1)
    df_full['final'] = df_full['new_customers'].fillna(0) + df_full['customer'].fillna(0)
    return df_full['final'].mean()
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.



# Reading Data
def read_data():
    print(f'Read data')
    train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')
    print(f"train shape: {train_df.shape}")
    test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')
    print(f"test shape: {test_df.shape}")
    train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
    print(f"train labels shape: {train_labels_df.shape}")
    specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
    print(f"specs shape: {specs_df.shape}")
    sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
    print(f"sample submission shape: {sample_submission_df.shape}")
    return train_df, test_df, train_labels_df, specs_df, sample_submission_df

# Assigning attempt result 
def attempt_result(df, res):
    result = np.where(((df['type'] == 'Assessment') &
                       (df['title'] != 'Bird Measurer (Assessment)') &
                       (df['event_code'] == 4100) &
                       (df['event_data'].str.contains('"correct":{}'.format(res)))) | 

                      ((df['type'] == 'Assessment') &
                       (df['title'] == 'Bird Measurer (Assessment)') &
                       (df['event_code'] == 4110) &
                       (df['event_data'].str.contains('"correct":{}'.format(res)))), 1, 0)
    return result

# Compute attempt results
def compute_attempt_result(df):
    df['att_succ'] = attempt_result(df, 'true')
    df['att_fail'] = attempt_result(df, 'false')
    return df

# Fix the titles
def fix_titles(df):
    df.title = df.title.str.replace(' (Activity)', '', regex=False)
    df.title = df.title.str.replace(' (Assessment)', '', regex=False)
    df.title = df.title.str.replace("-|,|'|!", '')
    df.title = df.title.str.replace(' ', '_', regex=False)
    return df

# Aggregating data by game sessions
def group_by_game_session(df):
    df_agg = df.groupby('game_session').agg(
        installation_id = pd.NamedAgg('installation_id', max),
        start_time = pd.NamedAgg('timestamp', min),
        world = pd.NamedAgg('world', max),
        type = pd.NamedAgg('type', max),
        title = pd.NamedAgg('title', max),
        event_count = pd.NamedAgg('event_count', max),
        game_time = pd.NamedAgg('game_time', max),
        num_correct = pd.NamedAgg('att_succ', sum),
        num_incorrect = pd.NamedAgg('att_fail', sum)
    )
    return df_agg

# Calculate accuracy
def calculate_accuracy(num_correct, num_incorrect):
    if (num_correct + num_incorrect) > 0:
        accuracy = num_correct / (num_correct + num_incorrect)
    else:
        accuracy = 0
    return accuracy

# Label accuracy group
def label_accuracy_group(num_correct, num_incorrect):

    if (num_correct == 1) & (num_incorrect == 0) :
        return 3
    elif  (num_correct == 1) & (num_incorrect == 1) :
        return 2
    elif  (num_correct == 1) & (num_incorrect >= 2) :
        return 1
    else :
        return 0
    
# Compute accuracy and accuracy group    
def compute_accuracy(df):
    df['accuracy'] = df.apply(lambda row : calculate_accuracy(row['num_correct'], row['num_incorrect']), axis=1)
    df['accuracy_group'] = df.apply(lambda row : label_accuracy_group(row['num_correct'], row['num_incorrect']), axis=1)
    return df

# Keep only installation ids with at least two sessions including at least one assessment
def remove_installations_with_no_assessments(df):
    return df.groupby(['installation_id']).filter(lambda x: (x[x['type'] == 'Assessment'].shape[0] >= 1)).reset_index(drop=True)

# Extract last assessment from each installation id
def extract_last_assessment(df, data_type):
    df_last = df[df.type == 'Assessment'].groupby('installation_id').apply(lambda x: x.sort_values('start_time'))
    df_last = df_last.reset_index(drop=True)
    df_last = df_last.groupby('installation_id').tail(1)
    if data_type == 'train':
        df_last = df_last[['installation_id', 'world', 'title', 'accuracy', 'accuracy_group']]
        df_last.columns = ['installation_id', 'target_world', 'target_title', 'target_accuracy', 'target_accuracy_group']
    else:
        df_last = df_last[['installation_id', 'world', 'title']]
        df_last.columns = ['installation_id', 'target_world', 'target_title']
    return df_last

# Delete sessions after last assessment from each installation id
def delete_after_last_assessment(df):
    df_sorted = df.groupby('installation_id').apply(lambda x: x.sort_values('start_time')).reset_index(drop=True)
    df_bl = df_sorted.groupby('installation_id').apply(lambda x: x.reset_index(drop=True).iloc[:x.reset_index(drop=True)[x.reset_index(drop=True)['type']=='Assessment'].last_valid_index()]).reset_index(drop=True)
    return df_bl

# Grouping game sessions by installation id to get the training data
def group_by_installation_id(df_by_session, data_type):
    
    # We first extract the last assessment for each installation id, this is the target we are trying predict the accuracy group for
    target_df = extract_last_assessment(df_by_session, data_type).set_index('installation_id')
    target_df = pd.get_dummies(target_df, columns=['target_world', 'target_title'])
    
    # We then delete all sessions that happened after the last assessment for each installation id
    df = delete_after_last_assessment(df_by_session)
    
    # Then we start aggregating data: 
    
    # How many sessions for each World/Type/Title
    df_agg_dummies = (pd.get_dummies(df[['installation_id', 'world', 'type', 'title']],
                                     columns=['world', 'type', 'title'])
                      .groupby(['installation_id'])
                      .sum()
                     )
    
    # Stats for event_count and game_time
    df_agg_numeric = (df[['installation_id', 'event_count', 'game_time']]
                      .groupby(['installation_id'])
                      .agg([np.sum, np.mean, np.std, np.min, np.max])
                     )
    
    df_agg_numeric.columns = ["_".join(x) for x in df_agg_numeric.columns.ravel()]
    
    # How many sessions for each accuracy_group
    df_agg_assessments_dummies = (pd.get_dummies(df[df.type == 'Assessment'][['installation_id', 'accuracy_group']],
                                                 columns=['accuracy_group'])
                                  .groupby(['installation_id'])
                                  .sum()
                                 )
    
    # Stats for num_correct, num_incorrect, accuracy
    df_agg_assessments_numeric = (df[df.type == 'Assessment'][['installation_id', 'num_correct', 'num_incorrect', 'accuracy']]
                                  .groupby(['installation_id'])
                                  .agg([np.sum, np.mean, np.std, np.min, np.max])
                                 )
    
    df_agg_assessments_numeric.columns = ["_".join(x) for x in df_agg_assessments_numeric.columns.ravel()]

    # Joining everything together
    df_agg = (df_agg_dummies
              .join(df_agg_numeric)
              .join(df_agg_assessments_numeric)
              .join(df_agg_assessments_dummies)
              .join(target_df, how='outer')
              .fillna(0)
             )
    
    return df_agg




def compile_data(df, df_type):
    # Calculate assessments attempts results
    df_compiled = compute_attempt_result(df)
    # Fix titles
    df_compiled = fix_titles(df_compiled)
    # Grouping data by game sessions
    df_compiled = group_by_game_session(df_compiled)
    # Calculate accuracy and accuracy groups
    df_compiled = compute_accuracy(df_compiled)
    # Remove installation ids with no assessments
    df_compiled = remove_installations_with_no_assessments(df_compiled)
    # Grouping data by installation id's
    df_compiled = group_by_installation_id(df_compiled, df_type)
    # Return result
    return df_compiled
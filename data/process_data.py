import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Description: This function loads the csv-files containing disaster response messages and labeled categories
                 Files will be loaded into pandas dataframes and merged using the 'id' column.

    Arguments:
        messages_filepath:  string Filepath to the message .csv-file
        categories_filepath:  string Filepath to the categories .csv-file

    Returns:
        merged pandas DataFrame
    """
    
    # ----   Load category data   ------
    df_cat = pd.read_csv(categories_filepath)
    cat_values = [{c.split('-')[0]: int(c.split('-')[1]) for c in r['categories'].split(';')} for i, r in df_cat.iterrows()]
    df_cat = pd.concat([df_cat, pd.DataFrame(cat_values)], axis=1)
    df_cat.drop(columns=['categories'], inplace=True)
    
    # ----   Load message data   ----
    df_messages = pd.read_csv(messages_filepath)
    
    # ----   merge datasets   --------
    df = df_cat.merge(df_messages, how='outer', on='id')
    
    return df


def clean_data(df):
    """
    Description: This function cleans the merged pandas DataFrame containing disaster response messages

    Arguments:
        df:  pandas DataFrame containing disaster response messages

    Returns:
        cleaned pandas DataFrame
    
    Cleaning steps:
        - NaN values:
            - Nan values are only present in 'original' column
            - if original message wasn't in english -> no imputation necessary, rows will not be dropped
            
        - Column 'child_alone':
            - Values are zero in all rows
            - Shouldn't cause an error, therefore column will not be dropped
            
        - Column 'related':
            - Column is a bit messed up
            - Figure eight source: https://www.figure-eight.com/dataset/combined-disaster-response-data/
            - According to column description, the column can contain two values (yes/no)
            - In provided dataset the column contains three values (0, 1, 2)
            - Running df.groupby(by='related').sum() shows that for related=2, all category columns are 0
            - Also related label is partly set incorrectly (e.g. 0 although some category columns are 1)
            - related label will be replaced by a self generated label using the rule: 0 if all category columns are zero, else 1
            
        - Dropping duplicates:
            - Full duplicates and duplicates of ID's will be dropped in two steps
    """
    
    # Fix column 'related'
    category_columns = df.drop(columns=['id', 'related', 'message', 'original', 'genre']).columns.tolist()
    df['related_new'] = df[category_columns].sum(axis=1).clip(upper=1)
    
    N_false_labels = len(df[df['related']!=df['related_new']])
    
    if N_false_labels > 0:
        print('In {:d} rows, the "related" label is set incorrectly.\nValues will be corrected'.format(N_false_labels))
        df.drop(columns='related', inplace=True)
        df.rename(columns={'related_new': 'related'}, inplace=True)
    else:
        df.drop(columns='related_new', inplace=True)
    
    
    # Drop duplicates (full rows)
    N_full_duplicates = len(df) - len(df.drop_duplicates())
    if N_full_duplicates > 0:
        print('Dropping {:d} rows with full duplicates'.format(N_full_duplicates))
        df.drop_duplicates(inplace=True)
    
    # Drop duplicates (ID's)
    N_id_duplicates = len(df) - len(df.drop_duplicates(subset='id'))
    if N_id_duplicates > 0:
        print('Dropping {:d} rows with duplicate ID but divergent labels'.format(N_id_duplicates))
        df.drop_duplicates(subset='id', inplace=True)
    
    return df


def save_data(df, database_filename):
    """ Stores pandas Dataframe into sqlite database """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('f8_disater_response_data', engine, if_exists='replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
import sys
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



def load_data(messages_filepath, categories_filepath):
    '''Loading datasets and creating a dataframe'''
    messages = pd.read_csv(messages_filepath) #loading first dataset
    categories = pd.read_csv(categories_filepath) #loading second dataset
    df = pd.merge(categories, messages, how='inner', on="id", sort=True,
                  suffixes=('_c', '_m'), copy=True, indicator=False,
                  validate=None) #concatenating both into dataframe
    return df #df


def clean_data(df):
    '''Cleaning and transforming our data'''
    categories = df['categories'].str.split(";", expand=True) # create a dataframe of the 36 individual category columns
    row = categories.iloc[0, :] #use first row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: pd.Series(x[:-2]))
    categories.columns = category_colnames.loc[:,0] #rename the columns of `categories`
    #convert category values to just numbers 0 or 1
    for column in categories:
        for value in column:
            categories[column] = [x.split('-')[-1] for x in categories.iloc[:, 0]]
    categories = categories.astype(int)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df.drop('categories', axis=1), categories], axis=1)
    df = df.drop_duplicates() #remove duplicates
    return df


def save_data(df, database_filepath):
    ''' save the clean dataset into an sqlite database'''
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql("master", engine, index=False)
    return database_filepath


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
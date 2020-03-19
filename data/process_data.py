# import libraries

import sys
import pandas as pd
import sqlite3

def load_data(messages_filepath, categories_filepath):
    """
    Load Data
    
    This function loads the messages and cateories data located at the provided file path. 
    A merged data frame is returned.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
        
    return messages.merge(categories, on = 'id')


def clean_data(df):
    """
    Clean Data
    
    This function cleanes the combined datasets:
        -Categories are extracted with each established as a column
        -Duplicates are Dropped
        -data issues are fixed (instances where "2" is present, when data should be binary)
    A merged data frame is returned.
    """    
    
    #split categories into its own table
    categories=df.categories.str.split(';',expand=True)
    row = categories.iloc[1,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    

    #items listed as 2 (should all be binary)
    categories.replace(2,1,inplace = True)
    
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)

    #merge categories data back to original df
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(subset = ['message'], inplace = True)
    return df

def save_data(df, database_filename):
    cxn = sqlite3.connect(database_filename)
    df.to_sql('messages_cleaned', cxn, if_exists='replace',index=False)
      

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
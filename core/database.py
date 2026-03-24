import pandas as pd
import os
from core.config import CSV_PATH

def load_database():
    """
    Load database.
    if the file doesn't exist, it creates an empty dataframe with the correct structure.
    """
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)


    return pd.DataFrame(columns=[
        "title",
        "description",
        "keywords",
        "category",
        "year"
    ])


def save_database(df):
    """
    Save the document database to CSV format.
    """
    df.to_csv(CSV_PATH, index=False)


def add_entry(df, entry):
    """
    Add a new entry (movie/document) to the DB.
    The entry must respect the structure :
    title, description, keywords, category, year
    """
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    save_database(df)
    return df


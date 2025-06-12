import pandas as pd

# Load and clean the dataset
def load_dataset(path):
    df = pd.read_excel(path)
    
    # Optional: clean column names (remove spaces, lower case, replace spaces with underscores)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    
    # Print columns to verify
    print("Dataset columns:", df.columns.tolist())
    
    return df
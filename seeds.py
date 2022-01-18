import pandas as pd
import ast

def get_seed(seed_id, file='gathered_seeds.json'):
    """Load dataframe from file, filter by seed_id and return seeds
    """
    seeds_df = pd.read_json(file, orient='records')
    item = seeds_df.loc[seeds_df['Seeds ID'] == seed_id]
    seeds = item['Seeds'].values[0]
    category = item['Category'].values[0]
    seed_list = ast.literal_eval(seeds)
    return seed_list, category

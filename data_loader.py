"""
Data Loader for Movies and TV Shows

Loads and combines movie and TV show data from multiple sources.
"""

import pandas as pd
import os
from typing import Optional


def load_movies_and_tv_shows(movies_file: str = 'data/movies.csv', 
                             tv_shows_file: Optional[str] = 'data/tv_shows.csv') -> pd.DataFrame:
    """
    Load movies and TV shows, combining them into a single DataFrame.
    
    Parameters:
    -----------
    movies_file : str
        Path to movies CSV file (MovieLens format)
    tv_shows_file : str, optional
        Path to TV shows CSV file. If None, only movies will be loaded.
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with columns: 'id', 'title', 'genres', 'type'
        where 'type' is 'movie' or 'tv_show'
    """
    all_content = []
    
    # Load movies
    if os.path.exists(movies_file):
        try:
            movies_df = pd.read_csv(movies_file)
            # Ensure required columns exist
            if 'movieId' in movies_df.columns and 'title' in movies_df.columns and 'genres' in movies_df.columns:
                movies_df = movies_df.rename(columns={'movieId': 'id'})
                movies_df['type'] = 'movie'
                movies_df = movies_df[['id', 'title', 'genres', 'type']]
                all_content.append(movies_df)
        except Exception as e:
            print(f"Warning: Could not load movies from {movies_file}: {e}")
    
    # Load TV shows
    if tv_shows_file and os.path.exists(tv_shows_file):
        try:
            tv_df = pd.read_csv(tv_shows_file)
            # Handle different possible column names
            if 'id' not in tv_df.columns and 'tvId' in tv_df.columns:
                tv_df = tv_df.rename(columns={'tvId': 'id'})
            if 'type' not in tv_df.columns:
                tv_df['type'] = 'tv_show'
            # Ensure required columns
            if 'id' in tv_df.columns and 'title' in tv_df.columns:
                if 'genres' not in tv_df.columns:
                    tv_df['genres'] = ''  # Add empty genres if missing
                tv_df = tv_df[['id', 'title', 'genres', 'type']]
                all_content.append(tv_df)
        except Exception as e:
            print(f"Warning: Could not load TV shows from {tv_shows_file}: {e}")
    
    if not all_content:
        return pd.DataFrame(columns=['id', 'title', 'genres', 'type'])
    
    # Combine all content
    combined_df = pd.concat(all_content, ignore_index=True)
    return combined_df


def create_sample_tv_shows_csv(output_file: str = 'data/tv_shows.csv'):
    """
    Create a sample TV shows CSV file with popular TV series.
    This is a placeholder - users should replace with their own TV show data.
    """
    sample_tv_shows = [
        {'id': 10001, 'title': 'Breaking Bad (2008)', 'genres': 'Crime|Drama|Thriller'},
        {'id': 10002, 'title': 'Game of Thrones (2011)', 'genres': 'Action|Adventure|Drama|Fantasy'},
        {'id': 10003, 'title': 'The Office (2005)', 'genres': 'Comedy'},
        {'id': 10004, 'title': 'Stranger Things (2016)', 'genres': 'Drama|Fantasy|Horror|Mystery|Sci-Fi|Thriller'},
        {'id': 10005, 'title': 'The Crown (2016)', 'genres': 'Biography|Drama|History'},
        {'id': 10006, 'title': 'The Mandalorian (2019)', 'genres': 'Action|Adventure|Drama|Fantasy|Sci-Fi|Western'},
        {'id': 10007, 'title': 'The Witcher (2019)', 'genres': 'Action|Adventure|Drama|Fantasy'},
        {'id': 10008, 'title': 'Chernobyl (2019)', 'genres': 'Drama|History|Thriller'},
        {'id': 10009, 'title': 'Fleabag (2016)', 'genres': 'Comedy|Drama'},
        {'id': 10010, 'title': 'The Boys (2019)', 'genres': 'Action|Comedy|Crime|Drama|Sci-Fi|Thriller'},
        {'id': 10011, 'title': 'Succession (2018)', 'genres': 'Comedy|Drama'},
        {'id': 10012, 'title': 'Better Call Saul (2015)', 'genres': 'Crime|Drama'},
        {'id': 10013, 'title': 'The Last of Us (2023)', 'genres': 'Action|Adventure|Drama|Horror|Sci-Fi|Thriller'},
        {'id': 10014, 'title': 'House of Cards (2013)', 'genres': 'Drama'},
        {'id': 10015, 'title': 'Black Mirror (2011)', 'genres': 'Drama|Mystery|Sci-Fi|Thriller'},
        {'id': 10016, 'title': 'The Handmaid\'s Tale (2017)', 'genres': 'Drama|Sci-Fi|Thriller'},
        {'id': 10017, 'title': 'Westworld (2016)', 'genres': 'Drama|Mystery|Sci-Fi|Thriller|Western'},
        {'id': 10018, 'title': 'True Detective (2014)', 'genres': 'Crime|Drama|Mystery|Thriller'},
        {'id': 10019, 'title': 'Fargo (2014)', 'genres': 'Crime|Drama|Thriller'},
        {'id': 10020, 'title': 'The Expanse (2015)', 'genres': 'Action|Adventure|Drama|Mystery|Sci-Fi'},
    ]
    
    df = pd.DataFrame(sample_tv_shows)
    df['type'] = 'tv_show'
    df.to_csv(output_file, index=False)
    print(f"Created sample TV shows file: {output_file}")
    return df


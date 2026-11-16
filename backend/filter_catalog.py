"""Build a quality-filtered TMDB catalog linked to MovieLens by TMDB ID.

MovieLens data: ratings.csv, links.csv (movies.csv optional).
TMDB data: TMDB_movie_dataset_v11.csv (override with --tmdb).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def clean_tmdb(df):
    required = {'id', 'title', 'overview', 'genres'}
    if missing := required - set(df):
        raise ValueError(f'TMDB missing columns: {sorted(missing)}')
    df = df.copy()
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df = df.dropna(subset=['id']).copy()
    df['id'] = df['id'].astype('int64')
    df = df.drop_duplicates('id')
    if 'status' in df:
        df = df[df['status'].fillna('').str.casefold() == 'released'].copy()
    if 'adult' in df:
        df = df[~df['adult'].fillna(False).astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])].copy()
    for col in ('title', 'overview', 'genres', 'keywords', 'tagline'):
        if col not in df:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str).str.strip()
    df = df[(df.title != '') & (df.genres != '') & (df.overview.str.len() >= 50)].copy()
    for col in ('vote_count', 'popularity', 'vote_average'):
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
    return df


def build_catalog(tmdb, ratings, links, size=25000, min_ratings=5):
    if size < 1 or min_ratings < 1:
        raise ValueError('size and min_ratings must be positive')
    movies = clean_tmdb(tmdb)
    link = links[['movieId', 'tmdbId']].copy()
    link['tmdbId'] = pd.to_numeric(link.tmdbId, errors='coerce')
    link = link.dropna(subset=['movieId', 'tmdbId']).copy()
    link['tmdbId'] = link.tmdbId.astype('int64')
    # Resolve rare ambiguous mappings deterministically, without multiplying rows.
    link = link.sort_values('movieId').drop_duplicates('tmdbId')
    counts = ratings.groupby('movieId').size().rename('rating_count')
    link = link.join(counts, on='movieId')
    movies = movies.merge(link, left_on='id', right_on='tmdbId', how='left', validate='one_to_one')
    movies['rating_count'] = movies.rating_count.fillna(0).astype('int64')
    collaborative = movies[movies.rating_count >= min_ratings].sort_values(
        ['rating_count', 'vote_count', 'id'], ascending=[False, False, True]
    ).head(size)
    remainder = max(0, size - len(collaborative))
    pool = movies.loc[~movies.id.isin(collaborative.id)].copy()
    pool['primary_genre'] = pool.genres.str.split(',').str[0].str.strip()
    pool = pool.sort_values(['vote_count', 'popularity', 'id'], ascending=[False, False, True])
    pool['genre_rank'] = pool.groupby('primary_genre').cumcount()
    extra = pool.sort_values(['genre_rank', 'vote_count', 'popularity', 'id'],
                             ascending=[True, False, False, True]).head(remainder)
    selected = pd.concat([collaborative, extra], ignore_index=True)
    selected = selected.drop(columns=['primary_genre', 'genre_rank', 'tmdbId'], errors='ignore')
    valid = selected.movieId.dropna().astype('int64')
    filtered = ratings[ratings.movieId.isin(valid)].copy()
    if selected.id.duplicated().any():
        raise AssertionError('duplicate TMDB IDs')
    return selected, filtered


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=Path('data/raw'))
    p.add_argument('--output', type=Path, default=Path('data/processed'))
    p.add_argument('--tmdb', default='TMDB_movie_dataset_v11.csv')
    p.add_argument('--size', type=int, default=25000)
    p.add_argument('--min-ratings', type=int, default=5)
    args = p.parse_args()
    fields = {'id', 'title', 'overview', 'genres', 'keywords', 'tagline', 'status', 'adult', 'vote_count', 'vote_average', 'popularity', 'release_date', 'poster_path'}
    tmdb = pd.read_csv(args.raw / args.tmdb, usecols=lambda c: c in fields, low_memory=False)
    ratings = pd.read_csv(args.raw / 'ratings.csv')
    links = pd.read_csv(args.raw / 'links.csv')
    movies, chosen_ratings = build_catalog(tmdb, ratings, links, args.size, args.min_ratings)
    args.output.mkdir(parents=True, exist_ok=True)
    movies.to_csv(args.output / 'training_movies.csv', index=False)
    chosen_ratings.to_csv(args.output / 'training_ratings.csv', index=False)
    print(f'Cleaned catalog selected: {len(movies):,} movies')
    print(f'MovieLens ratings retained: {len(chosen_ratings):,}/{len(ratings):,} ({len(chosen_ratings)/max(1,len(ratings)):.1%})')
    print(f'Movies with MovieLens ratings: {(movies.rating_count > 0).sum():,}')
    print(f'Users retained: {chosen_ratings.userId.nunique():,}')
    print(f'Output: {args.output.resolve()}')


if __name__ == '__main__':
    main()

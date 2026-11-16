"""Inference example. Additional ratings are keyed by TMDB IDs by default."""
import argparse
import joblib


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/hybrid.joblib')
    p.add_argument('--user', type=int, default=None, help='Existing MovieLens user ID')
    p.add_argument('--rate', action='append', default=[], metavar='TMDB_ID:RATING',
                   help='Add temporary rating, e.g. --rate 27205:5')
    p.add_argument('--top', type=int, default=10)
    args = p.parse_args()
    model = joblib.load(args.model)  # Trust the provenance of this file.
    additional = {}
    for pair in args.rate:
        mid, rating = pair.split(':', 1)
        additional[int(mid)] = float(rating)
    for movie in model.recommend(user_id=args.user, additional_ratings=additional, top_n=args.top):
        print(f"{movie['score']:.3f} | TMDB {movie['tmdb_id']:>7} | {movie['title']} | {movie['genres']}")


if __name__ == '__main__':
    main()

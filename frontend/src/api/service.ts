import {request} from './client';
export interface MovieData {
  tmdb_id: number;
  title: string;
  genres: string;
  overview: string;
  release_date: string;
  poster_path: string;
  vote_average: number | null;
  vote_count: number | null;
  score?: number;
  strategy?: string;
  genre_match?: number;
}
export interface UserInfo {user_id: number; username: string; rating_count: number}
export interface RatingData {tmdb_id: number; rating: number}
export const register = (username:string,password:string) => request<{username:string;user_id:number}>('/auth/register',{method:'POST',body:JSON.stringify({username,password})});
export const login = (username:string,password:string) => request<{username:string;user_id:number}>('/auth/login',{method:'POST',body:JSON.stringify({username,password})});
export const logout = () => request<void>('/auth/logout',{method:'POST'});
export const profile = () => request<UserInfo>('/users/me');
export const session = () => request<{user_id:number;username:string}>('/auth/session');
export const deleteAccount = () => request<void>('/users/me',{method:'DELETE'});
export const recommendations = (topN=50) => request<{items:MovieData[];recommendation_strategy:string;preferred_genres:string[]}>(`/recommendations?top_n=${topN}`);
export const searchMovies = (query:string) => request<{total:number;items:MovieData[]}>(`/movies?q=${encodeURIComponent(query)}&limit=40`);
export const movieDetails = (id:number) => request<MovieData>(`/movies/${id}`);
export const getRatings = () => request<RatingData[]>('/ratings');
export const saveRating = (tmdb_id:number,rating:number) => request('/ratings',{method:'PUT',body:JSON.stringify({tmdb_id,rating})});
export const getWatchlist = () => request<{items:MovieData[]}>('/watchlist');
export const addWatchlist = (tmdb_id:number) => request('/watchlist',{method:'POST',body:JSON.stringify({tmdb_id})});
export const removeWatchlist = (id:number) => request<void>(`/watchlist/${id}`,{method:'DELETE'});
export const getGenres = () => request<{genres:string[]}>('/genres');
export const getPreferences = () => request<{genres:string[]}>('/users/me/genres');
export const savePreferences = (genres:string[]) => request<{genres:string[]}>('/users/me/genres',{method:'PUT',body:JSON.stringify({genres})});
export const posterUrl = (path:string) => path && path !== 'nan' ? `https://image.tmdb.org/t/p/w500${path.startsWith('/') ? '' : '/'}${path}` : null;

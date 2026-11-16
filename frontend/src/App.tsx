import {useEffect, useState} from 'react';
import {register, login, logout, session, deleteAccount, recommendations, searchMovies, movieDetails, getRatings, saveRating, getWatchlist, addWatchlist, removeWatchlist, getGenres, getPreferences, savePreferences, posterUrl, type MovieData, type RatingData} from './api/service';

const A = '/assets';
type Screen = 'landing'|'signin'|'home'|'details'|'library'|'rate'|'search'|'genres';
type Movie = MovieData & {meta:string;genre:string};
const displayMovie=(m:MovieData):Movie=>({...m,meta:[m.release_date?.slice(0,4),m.genres?.split(',')[0]?.trim()].filter(Boolean).join(' · '),genre:m.genres||'Genre unavailable'});
function Icon({name,size}:{name:string;size?:number}) {return <img className="icon" src={`${A}/${name}.svg`} width={size} height={size} alt=""/>;}

function App() {
  const [screen,setScreen]=useState<Screen>('landing');
  const [authMode,setAuthMode]=useState<'login'|'register'>('login');
  const [menuOpen,setMenuOpen]=useState(false);
  const [member,setMember]=useState(false);
  const [busy,setBusy]=useState(false);
  const [message,setMessage]=useState('');
  const [movie,setMovie]=useState<Movie|null>(null);
  const [libraryTab,setLibraryTab]=useState<'want'|'watched'>('want');
  const [rating,setRating]=useState(4);
  const [allRecommendations,setAllRecommendations]=useState<Movie[]>([]);
  const [page,setPage]=useState(0);
  const [watchlist,setWatchlist]=useState<Movie[]>([]);
  const [ratings,setRatings]=useState<RatingData[]>([]);
  const [ratedMovies,setRatedMovies]=useState<(Movie&{rating:number})[]>([]);
  const [genres,setGenres]=useState<string[]>([]);
  const [selectedGenres,setSelectedGenres]=useState<string[]>([]);
  const [search,setSearch]=useState('');
  const [searchResults,setSearchResults]=useState<Movie[]>([]);
  const [username,setUsername]=useState('');
  const go=(next:Screen)=>{setScreen(next);setMenuOpen(false);setMessage('');window.scrollTo({top:0,behavior:'smooth'});};
  const chooseMovie=async (selected:Movie)=>{setMovie(selected);go('details');try{setMovie(displayMovie(await movieDetails(selected.tmdb_id)))}catch{/* keep summary */}};
  const fail=(error:unknown)=>setMessage(error instanceof Error?error.message:'An unexpected error occurred');
  const loadRecommendations=async()=>{const res=await recommendations(100);setAllRecommendations(res.items.map(displayMovie));setPage(0)};
  const loadLibrary=async()=>{
    const [watch,rs]=await Promise.all([getWatchlist(),getRatings()]);
    setWatchlist(watch.items.map(displayMovie));setRatings(rs);
    const details=await Promise.all(rs.map(async r=>{try{return {...displayMovie(await movieDetails(r.tmdb_id)),rating:r.rating}}catch{return null}}));
    setRatedMovies(details.filter((x):x is Movie&{rating:number}=>x!==null));
  };
  const loadGenres=async()=>{const [available,selected]=await Promise.all([getGenres(),getPreferences()]);setGenres(available.genres);setSelectedGenres(selected.genres)};
  useEffect(()=>{
    void session().then(u=>{setUsername(u.username);setMember(true);setScreen('home');return Promise.allSettled([loadRecommendations(),loadLibrary(),loadGenres()])}).catch(()=>{setMember(false);setScreen('landing')});
  },[]);
  const authenticate=async(user:string,password:string)=>{
    setBusy(true);setMessage('');try{
      const response=authMode==='login'?await login(user,password):await register(user,password);
      setUsername(response.username);setMember(true);
      go(authMode==='register'?'genres':'home');
      const loads=await Promise.allSettled([loadRecommendations(),loadLibrary(),loadGenres()]);
      const failed=loads.find(x=>x.status==='rejected');
      if(failed && failed.status==='rejected')fail(failed.reason);
    }catch(error){fail(error)}finally{setBusy(false)}
  };
  const signedOut=async()=>{try{await logout()}catch{}setMember(false);setAllRecommendations([]);setWatchlist([]);setRatedMovies([]);setUsername('');go('landing')};
  const removeAccount=async()=>{
    if(!window.confirm('Permanently delete your Reelwise account, ratings, and watchlist? This cannot be undone.'))return;
    try{await deleteAccount();setMember(false);setAllRecommendations([]);setWatchlist([]);setRatedMovies([]);setUsername('');go('landing')}catch(error){fail(error)}
  };
  const saveUserRating=async()=>{if(!movie)return;setBusy(true);try{await saveRating(movie.tmdb_id,rating);await Promise.all([loadLibrary(),loadRecommendations()]);setLibraryTab('watched');go('library')}catch(error){fail(error)}finally{setBusy(false)}};
  const saveToWatchlist=async()=>{if(!movie)return;setBusy(true);try{await addWatchlist(movie.tmdb_id);await loadLibrary();setLibraryTab('want');go('library')}catch(error){fail(error)}finally{setBusy(false)}};
  const removeFromWatchlist=async(id:number)=>{try{await removeWatchlist(id);await loadLibrary()}catch(error){fail(error)}};
  const find=async(query=search)=>{setBusy(true);setMessage('');try{const found=await searchMovies(query.trim());setSearchResults(found.items.map(displayMovie))}catch(error){fail(error)}finally{setBusy(false)}};
  const saveSelectedGenres=async()=>{setBusy(true);try{await savePreferences(selectedGenres);await loadRecommendations();go('home')}catch(error){fail(error)}finally{setBusy(false)}};
  const shown=allRecommendations.slice(page*10,(page+1)*10);
  return <div className="app">
    <Header open={menuOpen} onToggle={()=>setMenuOpen(x=>!x)} onBrand={()=>go(member?'home':'landing')} landingMark={screen==='landing'}/>
    <main>
      {message&&<p role="alert" className="api-error">{message}</p>}
      {screen==='landing'&&<Landing onSignIn={()=>{setAuthMode('login');go('signin')}} onSignUp={()=>{setAuthMode('register');go('signin')}}/>}
      {screen==='signin'&&<SignIn mode={authMode} onToggle={()=>{setAuthMode(authMode==='login'?'register':'login');setMessage('')}} onSubmit={authenticate} busy={busy}/>}
      {member&&screen==='home'&&<><p className="page" style={{paddingBottom:0}}>Welcome, {username}.</p><Home movies={shown} onReroll={()=>setPage(p=>(p+1)*10<allRecommendations.length?p+1:0)} onRandom={()=>{if(allRecommendations.length)void chooseMovie(allRecommendations[Math.floor(Math.random()*allRecommendations.length)])}} onMovie={m=>void chooseMovie(m)} onRate={()=>go('search')}/>{allRecommendations.length===0&&<p className="page">No recommendations available yet. Try rating movies or selecting genres.</p>}</>}
      {member&&screen==='details'&&movie&&<Details movie={movie} onBack={()=>go('home')} onWatchlist={()=>void saveToWatchlist()} onWatched={()=>{setRating(ratings.find(r=>r.tmdb_id===movie.tmdb_id)?.rating||4);go('rate')}}/>}
      {member&&screen==='library'&&<Library tab={libraryTab} onTab={setLibraryTab} onAdd={()=>go('search')} onRate={m=>{setMovie(m);setRating(ratings.find(r=>r.tmdb_id===m.tmdb_id)?.rating||4);go('rate')}} onMovie={m=>void chooseMovie(m)} wantMovies={watchlist} watchedMovies={ratedMovies} onRemove={id=>void removeFromWatchlist(id)}/>}
      {member&&screen==='rate'&&movie&&<RateMovie movie={movie} rating={rating} onRating={setRating} onBack={()=>go('library')} onSave={()=>void saveUserRating()}/>}
      {member&&screen==='search'&&<section className="page home"><div className="intro compact"><Eyebrow>FIND YOUR NEXT FAVORITE.</Eyebrow><h1>Explore movies.</h1></div><form className="search-form" onSubmit={e=>{e.preventDefault();if(search.trim())void find()}}><input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search by movie title" required/><button className="primary-action" disabled={busy}>Search</button></form><div className="poster-grid">{searchResults.map(m=><MovieCard key={m.tmdb_id} movie={m} onClick={()=>void chooseMovie(m)}/>)}</div></section>}
      {member&&screen==='genres'&&<section className="page home"><div className="intro compact"><Eyebrow>MAKE IT PERSONAL.</Eyebrow><h1>Your favorite genres.</h1><p>Choose genres you enjoy. You can change these anytime.</p></div><div className="genre-grid">{genres.map(g=><button key={g} type="button" className={selectedGenres.includes(g)?'genre-option selected':'genre-option'} onClick={()=>setSelectedGenres(a=>a.includes(g)?a.filter(x=>x!==g):[...a,g])}>{selectedGenres.includes(g)?'✓ ':''}{g}</button>)}</div><button className="primary-action" disabled={busy} onClick={()=>void saveSelectedGenres()}>Save genres</button></section>}
    </main><Footer/>
    {menuOpen&&<Menu member={member} current={screen} onGo={go} onSignOut={()=>void signedOut()} onDeleteAccount={()=>void removeAccount()}/>}
  </div>
}

function Header({
  open,
  onToggle,
  onBrand,
  landingMark,
}: {
  open: boolean;
  onToggle: () => void;
  onBrand: () => void;
  landingMark: boolean;
}) {
  return (
    <header className="site-header">
      <button className="brand" onClick={onBrand} aria-label="Reelwise home">
        <Icon name={landingMark ? "5b613" : "3d6d2"} size={24} />
        <span>reelwise</span>
      </button>
      <button
        className="menu-button"
        onClick={onToggle}
        aria-label={open ? "Close menu" : "Open menu"}
      >
        <Icon name={open ? "a8678" : "bb38c"} size={22} />
      </button>
    </header>
  );
}

function Footer() {
  return (
    <footer>
      <span>© 2026 Reelwise</span>
      <span>Made for movie people.</span>
    </footer>
  );
}

function Menu({
  member,
  current,
  onGo,
  onSignOut,
  onDeleteAccount,
}: {
  member: boolean;
  current: Screen;
  onGo: (screen: Screen) => void;
  onSignOut: () => void;
  onDeleteAccount: () => void;
}) {
  const links: { label: string; screen: Screen }[] = member
    ? [
        { label: "Recommendations", screen: "home" },
        { label: "My movies", screen: "library" },
        { label: "Find movies", screen: "search" },
        { label: "Favorite genres", screen: "genres" },
      ]
    : [{ label: "Home", screen: "landing" }];
  return (
    <div className="menu-layer">
      <button className="menu-backdrop" aria-label="Close menu" onClick={() => onGo(current)} />
      <nav className="menu-panel">
        <p className="menu-label">EXPLORE REELWISE</p>
        {links.map((link) => (
          <button
            key={link.label}
            className={current === link.screen ? "menu-link active" : "menu-link"}
            onClick={() => onGo(link.screen)}
          >
            <span>{link.label}</span>
            {current === link.screen ? (
              <small>YOU ARE HERE</small>
            ) : (
              <Icon name="538b7" size={18} />
            )}
          </button>
        ))}
        {!member && (
          <>
            <a className="menu-link" href="#faq">
              <span>FAQ</span>
              <Icon name="538b7" size={18} />
            </a>
            <a className="menu-link" href="mailto:hello@reelwise.example">
              <span>Contact</span>
              <Icon name="538b7" size={18} />
            </a>
          </>
        )}
        <div className="menu-rule" />
        {member ? (
          <div><button className="secondary-action" onClick={onSignOut}>
            Sign out <span>→</span>
          </button><button className="text-action" onClick={onDeleteAccount}>Delete account</button></div>
        ) : (
          <>
            <button className="primary-action" onClick={() => onGo("signin")}>
              Sign up <Icon name="87462" size={20} />
            </button>
            <button className="text-action" onClick={() => onGo("signin")}>
              Sign in
            </button>
          </>
        )}
      </nav>
    </div>
  );
}

function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <div className="eyebrow">
      <i />
      <span>{children}</span>
    </div>
  );
}

function Landing({ onSignIn, onSignUp }: { onSignIn: () => void; onSignUp: () => void }) {
  return (
    <section className="landing page">
      <div className="intro">
        <Eyebrow>LESS SEARCH. MORE CINEMA.</Eyebrow>
        <h1>
          Your taste.
          <br />
          Your next great
          <br />
          film.
        </h1>
        <p>Our AI finds movies for you based on your ratings and people who share your taste.</p>
      </div>
      <div className="rating-illustration" aria-label="Five star movie ratings">
        <div className="perforations">
          {Array.from({ length: 12 }, (_, i) => (
            <i key={i} />
          ))}
        </div>
        <div className="rating-concept">
          <strong>IT STARTS WITH YOUR TASTE</strong>
          <div className="outline-stars">
            {Array.from({ length: 5 }, (_, index) => (
              <Icon name="471af" size={24} key={index} />
            ))}
          </div>
          <span>Your ratings set the scene.</span>
        </div>
      </div>
      <div className="member-actions">
        <button className="primary-action" onClick={onSignUp}>
          Sign up <Icon name="87462" size={20} />
        </button>
        <p>
          Already a member? <button onClick={onSignIn}>Sign in</button>
        </p>
        <small>
          <Icon name="0a1e0" size={12} /> Sign in to see your recommendations.
        </small>
      </div>
    </section>
  );
}

function SignIn({ onSubmit, mode, onToggle, busy }: {onSubmit:(username:string,password:string)=>Promise<void>; mode:'login'|'register'; onToggle:()=>void; busy:boolean}) {
  const [username,setUsername]=useState('');
  const [password,setPassword]=useState('');
  const [showPassword,setShowPassword]=useState(false);
  return <section className="signin page">
    <div className="intro"><Eyebrow>BACK TO GREAT CINEMA.</Eyebrow><h1>{mode==='register'?'Join Reelwise.':'Welcome back.'}</h1><p>Recommendations shaped by your taste.</p></div>
    <form onSubmit={e=>{e.preventDefault(); void onSubmit(username,password)}}>
      <label>Username<input required value={username} onChange={e=>setUsername(e.target.value)} minLength={3} maxLength={40} autoComplete="username" /></label>
      <label>Password<span className="password-input"><input required minLength={8} value={password} onChange={e=>setPassword(e.target.value)} type={showPassword?'text':'password'} autoComplete={mode==='register'?'new-password':'current-password'}/><button type="button" aria-label="Show password" onClick={()=>setShowPassword(v=>!v)}><Icon name="3635f" size={20}/></button></span></label>
      <button className="primary-action" type="submit" disabled={busy}>{busy?'Working...': mode==='register'?'Create account':'Sign in'} <Icon name="975f5" size={20}/></button>
    </form><div className="member-actions"><p>{mode==='login'?'New to Reelwise?':'Already a member?'} <button onClick={onToggle}>{mode==='login'?'Sign up':'Sign in'}</button></p></div>
  </section>
}

function Home({
  movies,
  onReroll,
  onRandom,
  onMovie,
  onRate,
}: {
  movies: Movie[];
  onReroll: () => void;
  onRandom: () => void;
  onMovie: (movie: Movie) => void;
  onRate: () => void;
}) {
  return (
    <section className="home page">
      <div className="intro compact">
        <Eyebrow>YOUR PERSONAL BOX OFFICE.</Eyebrow>
        <h1>Picked for you.</h1>
        <p>AI picks based on your ratings and people who share your taste.</p>
      </div>
      <div className="discovery-actions">
        <button className="primary-action" onClick={onReroll}>
          Reroll recommendations <Icon name="abe35" size={20} />
        </button>
        <button className="secondary-action" onClick={onRandom}>
          Suggest a random movie <Icon name="b414a" size={20} />
        </button>
      </div>
      <div className="section-heading">
        <h2>Your recommendations</h2>
        <span>{movies.length} PICKS</span>
      </div>
      <div className="poster-grid">
        {movies.map((item) => (
          <MovieCard key={item.tmdb_id} movie={item} onClick={() => onMovie(item)} />
        ))}
      </div>
      <div className="refine">
        <span>More ratings. Better recommendations.</span>
        <button onClick={onRate}>Rate films</button>
      </div>
    </section>
  );
}

function Poster({ large = false, movie }: { large?: boolean; movie?:Movie }) {
  return (
    <div className={large ? "poster large" : "poster"}>
      {movie?.poster_path && posterUrl(movie.poster_path) ? <img className="real-poster" src={posterUrl(movie.poster_path)!} alt={`${movie.title} poster`} loading="lazy" /> : <><span>POSTER UNAVAILABLE</span><img src={`${A}/${large ? "9bf4a" : "1b373"}.svg`} alt="" /></>}
    </div>
  );
}

function MovieCard({ movie, onClick }: { movie: Movie; onClick: () => void }) {
  return (
    <button className="movie-card" onClick={onClick}>
      <Poster large movie={movie} />
      <strong>{movie.title}</strong>
      <small>{movie.meta}</small>
    </button>
  );
}

function Details({
  movie,
  onBack,
  onWatchlist,
  onWatched,
}: {
  movie: Movie;
  onBack: () => void;
  onWatchlist: () => void;
  onWatched: () => void;
}) {
  return (
    <section className="details page">
      <button className="back" onClick={onBack}>
        ← Back to recommendations
      </button>
      <div className="movie-identity">
        <span className="gold-label">PICKED FOR YOU</span>
        <Poster large movie={movie} />
        <h1>{movie.title}</h1>
        <p>{movie.meta.replace(/ · ([A-Za-z-]+) · /, " · ")}</p>
        <strong>{movie.genre}</strong>
      </div>
      <div className="overview">
        <h2>Overview</h2>
        <p>{movie.overview || 'No description is available for this movie.'}</p>
      </div>
      <div className="detail-actions">
        <button className="primary-action center" onClick={onWatchlist}>
          Add to watchlist
        </button>
        <button className="secondary-action center" onClick={onWatched}>
          Add to watched
        </button>
        <small>Watchlist saves to your Want to watch list.</small>
        <small>Already seen it? Add to Watched, then rate it.</small>
      </div>
    </section>
  );
}

function Library({
  tab,
  onTab,
  onAdd,
  onRate,
  onMovie,
  wantMovies,
  watchedMovies,
  onRemove,
}: {
  tab: "want" | "watched";
  onTab: (tab: "want" | "watched") => void;
  onAdd: () => void;
  onRate: (movie: Movie) => void;
  onMovie: (movie: Movie) => void;
  wantMovies: Movie[];
  watchedMovies: (Movie & {rating:number})[];
  onRemove:(id:number)=>void;
}) {
  return (
    <section className="library page">
      <div className="intro compact">
        <Eyebrow>YOUR LIFE IN FILM.</Eyebrow>
        <h1>My movies.</h1>
        <p>The films ahead. The ones that stayed.</p>
      </div>
      <button className="primary-action" onClick={onAdd}>
        Add movies <Icon name="37a19" size={20} />
      </button>
      <div className="tabs">
        <button className={tab === "want" ? "active" : ""} onClick={() => onTab("want")}>
          Want to watch <span>{wantMovies.length}</span>
        </button>
        <button className={tab === "watched" ? "active" : ""} onClick={() => onTab("watched")}>
          Watched <span>{watchedMovies.length}</span>
        </button>
      </div>
      <div className="list-heading">
        <span>{tab === "want" ? `${wantMovies.length} movies to look forward to` : `${watchedMovies.length} movies rated`}</span>
        <small>NEWEST FIRST</small>
      </div>
      <div className="movie-list">
        {tab === "want"
          ? wantMovies.map((item) => (
              <ListMovie key={item.tmdb_id} movie={item} onMovie={() => onMovie(item)}>
                <span className="gold-label">WANT TO WATCH</span>
                <button className="text-action" onClick={() => onRemove(item.tmdb_id)}>Remove</button>
                <button className="mark-button" onClick={() => onRate(item)}>
                  <Icon name="95b95" size={14} /> <span>Mark watched</span>
                </button>
              </ListMovie>
            ))
          : watchedMovies.map((item) => (
              <ListMovie key={item.tmdb_id} movie={item} onMovie={() => onMovie(item)}>
                <span className="gold-label">WATCHED</span>
                <small>Your rating</small>
                <div className="small-rating">
                  <AssetStars value={item.rating} /> <span>{item.rating}/5</span>{" "}
                  <button onClick={() => onRate(item)}>Edit</button>
                </div>
              </ListMovie>
            ))}
      </div>
      {(tab === 'want' ? wantMovies : watchedMovies).length === 0 && <p className="closing-note">No movies here yet. Search for a movie to get started.</p>}
      <p className="closing-note">
        {tab === "want"
          ? "Seen a film? Mark it watched, then rate it to help shape your next recommendations."
          : "More ratings. Better recommendations. Your ratings help shape future recommendations."}
      </p>
    </section>
  );
}

function ListMovie({
  movie,
  children,
  onMovie,
}: {
  movie: Movie;
  children?: React.ReactNode;
  onMovie?: () => void;
}) {
  return (
    <article className="list-movie">
      <button onClick={onMovie} aria-label={`Open ${movie.title}`}>
        <Poster movie={movie} />
      </button>
      <div className="list-movie-info">
        {children && Array.isArray(children) ? children[0] : null}
        <button className="movie-title" onClick={onMovie}>
          {movie.title}
        </button>
        <p>{movie.meta}</p>
        {children && Array.isArray(children) ? children.slice(1) : children}
      </div>
    </article>
  );
}

function Stars({ value }: { value: number }) {
  return (
    <span aria-label={`${value} out of 5 stars`}>
      {"★".repeat(value)}
      {"☆".repeat(5 - value)}
    </span>
  );
}

function AssetStars({ value }: { value: number }) {
  return (
    <span className="asset-stars" aria-label={`${value} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map((star) => (
        <Icon name={star <= value ? "d2869" : "ee55d"} size={17} key={star} />
      ))}
    </span>
  );
}

function RateMovie({
  movie,
  rating,
  onRating,
  onBack,
  onSave,
}: {
  movie: Movie;
  rating: number;
  onRating: (rating: number) => void;
  onBack: () => void;
  onSave: () => void;
}) {
  const descriptions = ["", "Not for me", "It was okay", "Liked it", "Really liked it", "Loved it"];
  return (
    <section className="rate page">
      <button className="back" onClick={onBack}>
        ← Back to my movies
      </button>
      <div className="rate-intro">
        <span className="gold-label">ADDED TO WATCHED</span>
        <h1>How was the movie?</h1>
        <p>Your next great recommendation starts with your rating.</p>
      </div>
      <div className="watched-movie">
        <Poster movie={movie} />
        <div>
          <h2>{movie.title}</h2>
          <p>{movie.meta}</p>
          <strong>In your Watched list</strong>
        </div>
      </div>
      <div className="rating-picker">
        <h3>Your rating</h3>
        <p>Tap a star to choose a rating.</p>
        <div className="rating-buttons">
          {[1, 2, 3, 4, 5].map((value) => (
            <button
              key={value}
              onClick={() => onRating(value)}
              aria-label={`Rate ${value} out of 5`}
            >
              {value <= rating ? "★" : "☆"}
            </button>
          ))}
        </div>
        <strong>
          {rating} / 5 — {descriptions[rating]}
        </strong>
      </div>
      <div className="rate-actions">
        <button className="primary-action center" onClick={onSave}>
          Save rating
        </button>
        <button className="text-action" onClick={onBack}>
          Rate later
        </button>
        <p>Save your rating to add this movie to Watched.</p>
      </div>
    </section>
  );
}


export default App;

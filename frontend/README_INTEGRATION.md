# Frontend integration
See `../README.md` for complete setup.

Frontend connections are centralized in `src/api/client.ts` and `src/api/service.ts`.
The main UI is `src/App.tsx`, adapted from the supplied Figma design.
The API base URL defaults to `/api`, proxied by Vite to FastAPI on `127.0.0.1:8000`. Keep `VITE_API_URL=/api` when accessing Reelwise from other devices on your LAN. Restart Vite after changing env settings.

Social demo sections are intentionally not shown until a real backend is implemented.

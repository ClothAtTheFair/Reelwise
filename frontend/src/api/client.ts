/** Same-origin HTTP client. Nginx (production) or Vite (development) proxies /api. */
export const API_URL = import.meta.env.VITE_API_URL || '/api';

function csrfToken(): string | null {
  const entry = document.cookie.split('; ').find(c => c.startsWith('reelwise_csrf='));
  return entry ? decodeURIComponent(entry.slice('reelwise_csrf='.length)) : null;
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body !== undefined) headers.set('Content-Type', 'application/json');
  if (init.method && !['GET', 'HEAD'].includes(init.method.toUpperCase())) {
    const csrf = csrfToken();
    if (csrf) headers.set('X-CSRF-Token', csrf);
  }
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      credentials: 'same-origin',
      headers,
    });
  } catch {
    throw new Error('Cannot connect to Reelwise. Please try again shortly.');
  }
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail;
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

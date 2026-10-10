// The Bonsai runtime owns the single service worker and the model cache.
// Workbox's second registration can reload a page while a local model is generating.
import { readable } from 'svelte/store';
export function usePwa() {
	return {
		get needRefresh() {
			return readable(false);
		},
		get needRefreshByStorage() {
			return false;
		},
		updateServiceWorker: async () => {
			window.location.reload();
		}
	};
}

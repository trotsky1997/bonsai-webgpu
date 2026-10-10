let promise: Promise<{ call: (request: unknown) => any }>;
export function core() {
	return (promise ??= import(
		/* @vite-ignore */ new URL('./web/ui-core.mjs?v=0.4.1', document.baseURI).href
	));
}

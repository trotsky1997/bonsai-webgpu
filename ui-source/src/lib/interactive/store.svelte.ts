import {normalizeUiArgs} from './normalize';
import { SvelteMap } from 'svelte/reactivity';
import { core } from './core';
import type { UiCalculation, UiDefinition, UiPatchResult, UiRecord, UiState } from './types';
const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v));
const stable = (v: any): string =>
	JSON.stringify(v, (_, x) =>
		x && typeof x === 'object' && !Array.isArray(x)
			? Object.fromEntries(
					Object.keys(x)
						.sort()
						.map((k) => [k, x[k]])
				)
			: x
	);
class InteractiveStore {
	records = new SvelteMap<string, UiRecord>();
	private dbPromise?: Promise<IDBDatabase>;
	private channel?: BroadcastChannel;
	private flushers = new Map<string, Set<() => Promise<void>>>();
	private pending = new Map<string, Set<string>>();
	private pendingFor(id: string) {
		let set = this.pending.get(id);
		if (!set) {
			try {
				set = new Set(JSON.parse(sessionStorage.getItem('bonsai-ui-pending:' + id) ?? '[]'));
			} catch {
				set = new Set();
			}
			this.pending.set(id, set);
		}
		return set;
	}
	private savePending(id: string) {
		sessionStorage.setItem('bonsai-ui-pending:' + id, JSON.stringify([...this.pendingFor(id)]));
	}
	private db() {
		if (!this.dbPromise) {
			const base = new URL('./', document.baseURI).pathname;
			this.channel = new BroadcastChannel('bonsai-ui.v1:' + base);
			this.channel.onmessage = (event) => {
				const key = event.data?.key;
				if (typeof key === 'string') this.read(key).catch(console.error);
			};
			this.dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
				const r = indexedDB.open('bonsai-ui.v1:' + base, 1);
				r.onupgradeneeded = () => {
					const s = r.result.createObjectStore('components', { keyPath: 'key' });
					s.createIndex('conversation', 'conversationId');
				};
				r.onsuccess = () => resolve(r.result);
				r.onerror = () => reject(r.error);
			});
		}
		return this.dbPromise;
	}
	async read(key: string) {
		const db = await this.db();
		return new Promise<UiRecord | undefined>((resolve, reject) => {
			const r = db.transaction('components').objectStore('components').get(key);
			r.onsuccess = () => {
				const value = r.result as UiRecord | undefined;
				if (value) this.records.set(key, value);
				else this.records.delete(key);
				resolve(value);
			};
			r.onerror = () => reject(r.error);
		});
	}
	async present(args: any, conversationId: string, callId: string) {
        args=normalizeUiArgs(args);
		if (!args.data || typeof args.data !== 'object' || Array.isArray(args.data))
			throw Error(
				'data: required object. Put all component fields inside data, e.g. data:{items:[{id:"a",label:"Item"}]}. Call present_ui again with corrected arguments.'
			);
		const api = await core();
		const definition: UiDefinition = { ...args, version: args.version ?? 1, id: args.id ?? callId };
		const checked = api.call({ op: 'validate', definition, state: args.state ?? {} });
		delete checked.definition.state;
		const key = conversationId + '/' + definition.id,
			db = await this.db();
		return new Promise<UiRecord>((resolve, reject) => {
			const tx = db.transaction('components', 'readwrite'),
				store = tx.objectStore('components'),
				get = store.get(key);
			let result: UiRecord, error: Error;
			get.onsuccess = () => {
				try {
					const old = get.result as UiRecord | undefined;
					if (old) {
						if (stable(old.definition) !== stable(checked.definition))
							throw Error('组件定义发布后不能修改；请使用新的 id');
						result = old;
						if (!old.calls.includes(callId)) {
							result = { ...old, calls: [...old.calls, callId] };
							store.put(result);
						}
					} else {
						result = {
							key,
							conversationId,
							calls: [callId],
							definition: checked.definition,
							version: 1,
							state: checked.state,
							history: [{ version: 1, state: checked.state }],
							operations: [],
							updatedAt: Date.now()
						};
						store.add(result);
					}
				} catch (e) {
					error = e as Error;
					tx.abort();
				}
			};
			tx.oncomplete = () => {
				this.records.set(key, result);
				this.channel?.postMessage({ key });
				resolve(result);
			};
			tx.onabort = tx.onerror = () => reject(error ?? tx.error ?? Error('保存组件失败'));
		});
	}
	async patch(
		key: string,
		baseVersion: number,
		patch: UiState,
		operationId: string = crypto.randomUUID()
	): Promise<UiPatchResult> {
		const api = await core(),
			db = await this.db();
		return new Promise((resolve, reject) => {
			const tx = db.transaction('components', 'readwrite'),
				store = tx.objectStore('components'),
				get = store.get(key);
			let result: UiPatchResult, error: Error;
			get.onsuccess = () => {
				try {
					const current = get.result as UiRecord | undefined;
					if (!current) throw Error('组件不存在');
					if (current.operations.includes(operationId)) {
						result = { status: 'ok', record: current };
						return;
					}
					if (baseVersion > current.version) throw Error('状态版本超前');
					const base = current.history.find((s) => s.version === baseVersion);
					if (!base) {
						result = { status: 'conflict', record: current, fields: Object.keys(patch) };
						return;
					}
					const merged = api.call({
						op: 'merge',
						definition: current.definition,
						base: base.state,
						current: current.state,
						patch
					});
					if (merged.status === 'conflict') {
						result = { status: 'conflict', record: current, fields: merged.fields };
						return;
					}
					const changed = stable(merged.state) !== stable(current.state);
					const version = current.version + (changed ? 1 : 0);
					const record = {
						...current,
						state: merged.state,
						version,
						updatedAt: Date.now(),
						operations: [...current.operations, operationId].slice(-128),
						history: changed
							? [...current.history, { version, state: clone(merged.state) }].slice(-64)
							: current.history
					};
					store.put(record);
					result = { status: 'ok', record };
				} catch (e) {
					error = e as Error;
					tx.abort();
				}
			};
			tx.oncomplete = () => {
				this.records.set(key, result.record);
				this.channel?.postMessage({ key });
				resolve(result);
			};
			tx.onabort = tx.onerror = () => reject(error ?? tx.error ?? Error('保存状态失败'));
		});
	}
	async calculate(definition: UiDefinition, state: UiState): Promise<UiCalculation> {
		return (await core()).call({ op: 'evaluate', definition, state });
	}
	register(key: string, flush: () => Promise<void>) {
		let set = this.flushers.get(key);
		if (!set) this.flushers.set(key, (set = new Set()));
		set.add(flush);
		return () => {
			set?.delete(flush);
			if (!set?.size) this.flushers.delete(key);
		};
	}
	async flush(key: string) {
		for (const f of this.flushers.get(key) ?? []) await f();
	}
	async intent(record: UiRecord, intent: string) {
		await this.flush(record.key);
		let set = this.pendingFor(record.conversationId);
		if (!set) this.pending.set(record.conversationId, (set = new Set()));
		set.add(record.key);
		this.savePending(record.conversationId);
		window.dispatchEvent(
			new CustomEvent('bonsai-ui-intent', {
				detail: { conversationId: record.conversationId, intent, title: record.definition.title }
			})
		);
	}
	async bind(conversationId: string) {
		const keys = [...this.pendingFor(conversationId)];
		if (!keys.length) return '';
		for (const key of keys) await this.flush(key);
		const db = await this.db();
		const records = await new Promise<UiRecord[]>((resolve, reject) => {
			const tx = db.transaction('components'),
				store = tx.objectStore('components'),
				out: UiRecord[] = [];
			for (const key of keys) {
				const r = store.get(key);
				r.onsuccess = () => {
					if (r.result) out.push(r.result);
				};
			}
			tx.oncomplete = () => resolve(out);
			tx.onerror = () => reject(tx.error);
		});
		const contexts = await Promise.all(
			records.map(async (r) => ({
				id: r.definition.id,
				version: r.version,
				type: r.definition.type,
				title: r.definition.title,
				summary: r.definition.summary,
				state: r.state,
				data: r.definition.data,
				outputs: (await this.calculate(r.definition, r.state)).outputs
			}))
		);
		return '\n\n[interactive-ui.v1]\n' + JSON.stringify(contexts) + '\n[/interactive-ui.v1]';
	}
	acknowledge(conversationId: string) {
		this.pending.delete(conversationId);
		sessionStorage.removeItem('bonsai-ui-pending:' + conversationId);
	}
	async execute(name: string, args: any, conversationId: string, callId: string) {
		try {
			if(args.__protocol_error)throw Error(args.__protocol_error);
			if (name === 'present_ui') {
				const r = await this.present(args, conversationId, callId);
				return {
					content: JSON.stringify({
						ui_id: r.definition.id,
						version: r.version,
						summary: r.definition.summary,
						state: r.state,
						displayed: true
					}),
					isError: false
				};
			}
			const key = conversationId + '/' + args.id;
			if (!Number.isInteger(args.base_version) || args.base_version < 1)
				throw Error('base_version must be a positive integer');
			const result = await this.patch(key, args.base_version, args.patch, callId);
			return {
				content: JSON.stringify({
					ui_id: args.id,
					...result,
					status: result.status,
					record: undefined,
					state: result.record.state,
					version: result.record.version
				}),
				isError: result.status === 'conflict'
			};
		} catch (error) {
			return { content: 'UI error: ' + String((error as Error)?.message ?? error), isError: true };
		}
	}
	async retainCalls(conversationId: string, liveIds: Set<string>) {
		const db=await this.db();const changed:string[]=[];
		await new Promise<void>((resolve,reject)=>{
			const tx=db.transaction('components','readwrite');const request=tx.objectStore('components').index('conversation').openCursor(IDBKeyRange.only(conversationId));
			request.onsuccess=()=>{const c=request.result;if(!c)return;const r=c.value as UiRecord;const calls=r.calls.filter(id=>liveIds.has(id));if(!calls.length){c.delete();changed.push(r.key);}else if(calls.length!==r.calls.length){c.update({...r,calls});changed.push(r.key);}c.continue();};
			tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error);
		});
		for(const key of changed){await this.read(key);this.channel?.postMessage({key});}
	}
	async deleteConversation(conversationId: string) {
		const db = await this.db();
		await new Promise<void>((resolve, reject) => {
			const tx = db.transaction('components', 'readwrite'),
				index = tx.objectStore('components').index('conversation'),
				r = index.openCursor(IDBKeyRange.only(conversationId));
			r.onsuccess = () => {
				const cursor = r.result;
				if (cursor) {
					this.records.delete(String(cursor.primaryKey));
					cursor.delete();
					cursor.continue();
				}
			};
			tx.oncomplete = () => resolve();
			tx.onerror = () => reject(tx.error);
		});
		this.pending.delete(conversationId);
		sessionStorage.removeItem('bonsai-ui-pending:' + conversationId);
	}
}
export const interactiveStore = new InteractiveStore();

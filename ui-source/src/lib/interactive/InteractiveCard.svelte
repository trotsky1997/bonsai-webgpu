<script lang="ts">
	import { onMount } from 'svelte';
	import Chart from './Chart.svelte';
	import { interactiveStore } from './store.svelte';
	import type { UiCalculation, UiRecord, UiState } from './types';
	let { args, callId, conversationId }: { args: any; callId: string; conversationId: string } =
		$props();
	let key = $state(''),
		error = $state(''),
		status = $state('loading'),
		draft = $state<UiState>({}),
		baseVersion = $state(0),
		calculation = $state<UiCalculation>({ outputs: [], chart: null }),
		calcError = $state('');
	const record = $derived(interactiveStore.records.get(key));
	const definition = $derived(record?.definition);
	const data = $derived(definition?.data ?? {});
	const viewState = $derived({ ...record?.state, ...draft });
	const groups = $derived([
		...new Set((data.items ?? []).map((i: any) => i.group ?? i.phase ?? ''))
	] as string[]);
	const filtered = $derived(
		(data.rows ?? []).filter((r: any) => {
			const filter = String(viewState.filter ?? '').toLowerCase();
			return (
				!filter || [r.label, ...Object.values(r.values)].join(' ').toLowerCase().includes(filter)
			);
		})
	);
	const checked = $derived(
		(data.items ?? []).filter((i: any) => viewState['checked:' + i.id]).length
	);
	let timer: ReturnType<typeof setTimeout>,
		saving: Promise<void> | undefined,
		unregister: () => void = () => {},
		calculationSerial = 0;
	const storageKey = () => 'bonsai-ui-draft:' + key;
	onMount(() => {
		let live = true;
		interactiveStore
			.present(args, conversationId, callId)
			.then((r) => {
				if (!live) return;
				key = r.key;
				status = 'saved';
				try {
					const saved = JSON.parse(sessionStorage.getItem(storageKey()) ?? 'null');
					if (saved) {
						draft = saved.patch;
						baseVersion = saved.baseVersion;
						status = 'pending';
						timer = setTimeout(() => save().catch(() => {}), 300);
					}
				} catch {}
				unregister = interactiveStore.register(key, async () => {
					await save();
					if (status === 'conflict' || status === 'error')
						throw Error('请先解决组件状态冲突或保存错误');
				});
			})
			.catch((e) => {
				if (live) {
					error = String(e.message ?? e);
					status = 'error';
				}
			});
		return () => {
			live = false;
			clearTimeout(timer);
			unregister();
			if (Object.keys(draft).length) save().catch(() => {});
		};
	});
	$effect(() => {
		if (definition) {
			const current = viewState,
				serial = ++calculationSerial;
			interactiveStore
				.calculate(definition, current)
				.then((result) => {
					if (serial === calculationSerial) {
						calculation = result;
						calcError = '';
					}
				})
				.catch((e) => {
					if (serial === calculationSerial) calcError = String(e.message ?? e);
				});
		}
	});
	function set(keyName: string, value: string | number | boolean, immediate = false) {
		if (!record) return;
		if (!Object.keys(draft).length) baseVersion = record.version;
		draft = { ...draft, [keyName]: value };
		status = 'pending';
		error = '';
		sessionStorage.setItem(storageKey(), JSON.stringify({ baseVersion, patch: draft }));
		clearTimeout(timer);
		if (immediate) save().catch(() => {});
		else timer = setTimeout(() => save().catch(() => {}), 300);
	}
	async function save(): Promise<void> {
		clearTimeout(timer);
		if (saving) {
			await saving;
			if (Object.keys(draft).length && status !== 'conflict') return save();
			return;
		}
		if (!record || !Object.keys(draft).length || status === 'conflict') return;
		const patch = { ...draft };
		status = 'saving';
		saving = (async () => {
			try {
				const result = await interactiveStore.patch(key, baseVersion, patch);
				if (result.status === 'conflict') {
					status = 'conflict';
					error = '这些字段已在另一标签页修改：' + result.fields?.join('、');
					return;
				}
				const remaining = { ...draft };
				for (const k of Object.keys(patch)) if (remaining[k] === patch[k]) delete remaining[k];
				draft = remaining;
				baseVersion = result.record.version;
				status = Object.keys(draft).length ? 'pending' : 'saved';
				if (status === 'saved') sessionStorage.removeItem(storageKey());
				else sessionStorage.setItem(storageKey(), JSON.stringify({ baseVersion, patch: draft }));
			} catch (e) {
				status = 'error';
				error = String((e as Error).message);
				throw e;
			} finally {
				saving = undefined;
			}
		})();
		await saving;
		if (Object.keys(draft).length && status === 'pending') await save();
	}
	function useCurrent() {
		draft = {};
		status = 'saved';
		error = '';
		sessionStorage.removeItem(storageKey());
	}
	async function reapply() {
		const current = await interactiveStore.read(key);
		if (current) {
			baseVersion = current.version;
			status = 'pending';
			await save();
		}
	}
	async function ask() {
		if (!record) return;
		try {
			await interactiveStore.intent(record, '请根据这个组件的当前选择继续分析。');
		} catch (e) {
			error = String((e as Error).message);
		}
	}
	async function copy() {
		if (!record) return;
		try {
			await save();
			if (status === 'conflict') throw Error('请先解决状态冲突');
			const current = await interactiveStore.read(key);
			await navigator.clipboard.writeText(
				JSON.stringify(
					{
						protocol: 'interactive-ui.v1',
						definition: current?.definition,
						state: current?.state,
						version: current?.version
					},
					null,
					2
				)
			);
			error = '已复制组件与状态快照';
		} catch (e) {
			error = String((e as Error).message);
		}
	}
	function number(event: Event, c: any) {
		const value = (event.target as HTMLInputElement).valueAsNumber;
		if (Number.isFinite(value))
			set('param:' + c.id, Math.max(c.min ?? 0, Math.min(c.max ?? 100, value)));
	}
	function node(id: string) {
		return data.nodes.find((n: any) => n.id === id);
	}
	const nx = (n: any) => 50 + n.x * 5,
		ny = (n: any) => 40 + n.y * 2.1;
</script>

<section
	aria-label={definition?.title ?? args.title ?? '交互组件'}
	class="ui-card"
	data-ui-id={definition?.id}
	data-ui-kind={definition?.type}
	data-ui-version={record?.version}
>
	<header>
		<span aria-hidden="true" class="mark">◈</span>
		<div>
			<h3>{definition?.title ?? args.title ?? '正在校验组件'}</h3>
			<p>{definition?.summary ?? args.summary ?? ''}</p>
		</div>
		<span class="badge">交互</span>
	</header>
	{#if definition && record}
		<div class="body">
			{#if definition.type === 'comparison'}
				<label class="filter"
					>筛选方案<input
						aria-label="筛选方案"
						oninput={(e) => set('filter', e.currentTarget.value)}
						placeholder="名称或属性值"
						type="search"
						value={viewState.filter ?? ''}
					/></label
				>
				<div aria-label="方案对比表" class="scroll" role="region">
					<table>
						<thead
							><tr
								><th scope="col">方案</th>{#each data.columns as col}<th scope="col">{col.label}</th
									>{/each}<th scope="col">选择</th></tr
							></thead
						><tbody
							>{#each filtered as row (row.id)}<tr class:chosen={viewState.selected === row.id}
									><th scope="row">{row.label}</th>{#each data.columns as col}<td
											>{row.values[col.key] ?? '—'}</td
										>{/each}<td
										><button
											aria-pressed={viewState.selected === row.id}
											onclick={() => set('selected', row.id, true)}
											type="button">{viewState.selected === row.id ? '已选择' : '选择'}</button
										></td
									></tr
								>{/each}</tbody
						>
					</table>
					{#if !filtered.length}<p>没有匹配的方案。</p>{/if}
				</div>
			{:else if definition.type === 'checklist'}
				<div aria-live="polite" class="progress-label">
					已完成 {checked} / {data.items.length}<span
						>{Math.round((checked / data.items.length) * 100)}%</span
					>
				</div>
				<progress aria-label="清单完成进度" max={data.items.length} value={checked}></progress>
				<div class="scroll list">
					{#each groups as group}{#if group}<h4>
								{group}
							</h4>{/if}{#each data.items.filter((i: any) => (i.group ?? i.phase ?? '') === group) as item (item.id)}<label
								class="check"
								><input
									checked={!!viewState['checked:' + item.id]}
									onchange={(e) => set('checked:' + item.id, e.currentTarget.checked, true)}
									type="checkbox"
								/><span
									><strong>{item.label}</strong>{#if item.description}<small
											>{item.description}</small
										>{/if}</span
								></label
							>{/each}{/each}
				</div>
				{#if data.options}<label class="filter"
						>选项<select
							onchange={(e) => set('option', e.currentTarget.value, true)}
							value={viewState.option ?? ''}
							><option value="">请选择</option>{#each data.options as option}<option
									value={option.id}>{option.label}</option
								>{/each}</select
						></label
					>{/if}
			{:else if definition.type === 'timeline'}
				<ol class="scroll timeline">
					{#each data.items as item (item.id)}<li class:chosen={viewState.selected === item.id}>
							<div class="time-head">
								<span>{item.date ?? item.phase ?? ''}</span><button
									aria-pressed={viewState.selected === item.id}
									onclick={() => set('selected', item.id, true)}
									type="button">{item.label}</button
								>{#if item.status}<small>{item.status}</small>{/if}
							</div>
							{#if item.description}<details>
									<summary>展开详情</summary>
									<p>{item.description}</p>
								</details>{/if}
						</li>{/each}
				</ol>
			{:else if definition.type === 'chart'}<Chart
					{data}
					onVisible={(k, v) => set(k, v, true)}
					{viewState}
				/>
			{:else if definition.type === 'parameters'}
				<div class="controls">
					{#each data.controls as c (c.id)}
						<label class="control"
							><span>{c.label}</span>
							{#if c.kind === 'switch'}<input
									aria-label={c.label}
									checked={!!viewState['param:' + c.id]}
									onchange={(e) => set('param:' + c.id, e.currentTarget.checked, true)}
									role="switch"
									type="checkbox"
								/>
							{:else if c.kind === 'select'}<select
									aria-label={c.label}
									onchange={(e) => set('param:' + c.id, e.currentTarget.value, true)}
									value={viewState['param:' + c.id]}
									>{#each c.options as option}<option value={option.value}>{option.label}</option
										>{/each}</select
								>
							{:else}<div class="numeric">
									{#if c.kind === 'slider'}<input
											aria-label={c.label + ' 滑块'}
											max={c.max ?? 100}
											min={c.min ?? 0}
											onchange={() => save().catch(() => {})}
											oninput={(e) => number(e, c)}
											step={c.step ?? 1}
											type="range"
											value={viewState['param:' + c.id]}
										/>{/if}<input
										aria-label={c.label + ' 数值'}
										max={c.max ?? 100}
										min={c.min ?? 0}
										onchange={() => save().catch(() => {})}
										oninput={(e) => number(e, c)}
										step={c.step ?? 1}
										type="number"
										value={viewState['param:' + c.id]}
									/>
								</div>{/if}
						</label>
					{/each}
				</div>
				{#if calcError}<p class="error" role="alert">计算：{calcError}</p>{/if}
				<div aria-live="polite" class="outputs">
					{#each calculation.outputs as output (output.id)}<div>
							<small>{output.label}</small><strong
								>{typeof output.value === 'number'
									? new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format(
											output.value
										)
									: String(output.value)}
								{output.unit ?? ''}</strong
							>
						</div>{/each}
				</div>
				{#if calculation.chart}<Chart data={calculation.chart} />{/if}
			{:else if definition.type === 'diagram'}
				<svg aria-label={definition.summary} class="diagram" role="img" viewBox="0 0 600 300"
					><title>{definition.title}</title><desc
						>{definition.summary}。下方部件按钮支持键盘选择。</desc
					>
					{#each data.edges as edge}{@const from = node(edge.from)}{@const to = node(edge.to)}<line
							x1={nx(from)}
							x2={nx(to)}
							y1={ny(from)}
							y2={ny(to)}
						/>{#if edge.label}<text
								text-anchor="middle"
								x={(nx(from) + nx(to)) / 2}
								y={(ny(from) + ny(to)) / 2 - 8}>{edge.label}</text
							>{/if}{/each}
					{#each data.nodes as n (n.id)}<g
							role="button"
							tabindex="0"
							aria-label={n.label}
							onkeydown={(e) => {
								if (e.key === 'Enter' || e.key === ' ') {
									e.preventDefault();
									set('selected', n.id, true);
								}
							}}
							class:selected={viewState.selected === n.id}
							onclick={() => set('selected', n.id, true)}
							><rect height="56" rx="12" width="112" x={nx(n) - 56} y={ny(n) - 28} /><text
								text-anchor="middle"
								x={nx(n)}
								y={ny(n) + 5}>{n.label.length > 15 ? n.label.slice(0, 15) + '…' : n.label}</text
							><title>{n.label}</title></g
						>{/each}
				</svg>
				<div aria-label="选择部件" class="node-buttons">
					{#each data.nodes as n (n.id)}<button
							aria-pressed={viewState.selected === n.id}
							onclick={() => set('selected', n.id, true)}
							type="button">{n.label}</button
						>{/each}
				</div>
				{#if viewState.selected}{@const selected = node(String(viewState.selected))}
					<div aria-live="polite" class="node-detail">
						<strong>{selected?.label}</strong>
						<p>{selected?.description ?? '已选择此部件。'}</p>
					</div>{/if}
			{/if}
		</div>
		<footer>
			<span
				aria-live="polite"
				class:unsaved={status !== 'saved'}
				title="状态仅保存在本浏览器，并同步到同源标签页"
				>{(
					{
						loading: '校验中',
						pending: '待保存',
						saving: '保存中',
						saved: '已在本浏览器保存',
						conflict: '状态冲突',
						error: '保存失败'
					} as Record<string, string>
				)[status]} · v{record.version}</span
			>
			<div>
				<button onclick={copy} type="button">复制快照</button><button
					class="primary"
					onclick={ask}
					type="button">带选择继续聊</button
				>
			</div>
		</footer>
	{/if}
	{#if error}<div
			class="error"
			role={status === 'error' || status === 'conflict' ? 'alert' : 'status'}
		>
			{error}{#if status === 'conflict'}<div class="conflict">
					<button onclick={useCurrent} type="button">采用当前值</button><button
						onclick={() => reapply().catch((e) => (error = e.message))}
						type="button">重新应用我的修改</button
					>
				</div>{/if}
		</div>{/if}
</section>

<style>
	.ui-card {
		width: 100%;
		min-width: 0;
		max-width: 48rem;
		border: 1px solid var(--border, #cdd9da);
		border-radius: 18px;
		background: color-mix(in srgb, var(--background, #fff) 96%, #0f8b8d 4%);
		overflow: hidden;
		margin: 10px 0;
		color: var(--foreground, #162b30);
		font: 14px/1.5 inherit;
		container-type: inline-size;
	}
	.ui-card header {
		display: flex;
		gap: 12px;
		padding: 18px;
		align-items: flex-start;
	}
	.mark {
		background: #0b7d88;
		color: white;
		width: 32px;
		height: 32px;
		flex: none;
		display: grid;
		place-items: center;
		border-radius: 10px;
		font-size: 20px;
	}
	.ui-card header > div {
		min-width: 0;
		flex: 1;
	}
	.ui-card h3 {
		font-size: 16px;
		font-weight: 650;
		line-height: 1.4;
		margin: 0;
	}
	.ui-card header p {
		font-size: 12px;
		opacity: 0.75;
		margin: 5px 0 0;
		overflow-wrap: anywhere;
	}
	.badge {
		font-size: 10px;
		letter-spacing: 0.04em;
		padding: 3px 7px;
		background: color-mix(in srgb, #087f8c 10%, transparent);
		border-radius: 20px;
		flex: none;
	}
	.body {
		padding: 0 18px 14px;
		min-width: 0;
	}
	.scroll {
		max-height: 380px;
		overflow: auto;
		max-width: 100%;
		overscroll-behavior: contain;
	}
	.filter {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-bottom: 12px;
		font-size: 12px;
	}
	.filter input,
	.filter select {
		flex: 1;
		min-width: 0;
	}
	.ui-card button,
	.ui-card select,
	.ui-card input[type='number'],
	.ui-card input[type='search'] {
		min-height: 44px;
		border: 1px solid var(--border, #ccd8d9);
		background: var(--background, #fff);
		border-radius: 9px;
		padding: 9px 12px;
		font: inherit;
		color: inherit;
	}
	.ui-card button {
		cursor: pointer;
		min-width: 44px;
	}
	.ui-card button[aria-pressed='true'],
	.chosen {
		background: color-mix(in srgb, #087f8c 12%, transparent);
	}
	.ui-card button[aria-pressed='true'] {
		border-color: #087f8c;
	}
	.ui-card :focus-visible {
		outline: 2px solid #078596;
		outline-offset: 2px;
	}
	table {
		border-collapse: collapse;
		width: 100%;
		font-size: 13px;
	}
	th,
	td {
		padding: 10px 12px;
		text-align: left;
		border-bottom: 1px solid var(--border, #ccd8d9);
		white-space: nowrap;
	}
	th {
		font-weight: 600;
	}
	thead {
		position: sticky;
		top: 0;
		background: var(--background, #fff);
	}
	footer {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 10px;
		padding: 12px 18px;
		border-top: 1px solid var(--border, #ccd8d9);
		flex-wrap: wrap;
	}
	footer > span {
		font-size: 11px;
		opacity: 0.65;
	}
	footer > div {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
	}
	.ui-card button.primary {
		background: #087f8c;
		color: #fff;
		border-color: #087f8c;
	}
	.unsaved {
		color: #a15b00;
	}
	.error {
		font-size: 12px;
		padding: 10px 18px;
		overflow-wrap: anywhere;
		background: color-mix(in srgb, #c93d3d 7%, transparent);
	}
	.conflict {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
		margin-top: 8px;
	}
	.check {
		display: flex;
		min-height: 52px;
		gap: 12px;
		align-items: center;
		padding: 9px 6px;
		cursor: pointer;
	}
	.check input {
		width: 21px;
		height: 21px;
		accent-color: #087f8c;
		flex: none;
	}
	.check span {
		min-width: 0;
	}
	.check strong {
		font-weight: 500;
	}
	.check small {
		display: block;
		opacity: 0.7;
		overflow-wrap: anywhere;
	}
	.list h4 {
		font-size: 12px;
		opacity: 0.7;
		margin: 16px 0 5px;
	}
	.progress-label {
		display: flex;
		justify-content: space-between;
		font-size: 12px;
	}
	progress {
		width: 100%;
		height: 7px;
		accent-color: #087f8c;
		margin: 8px 0 12px;
	}
	.timeline {
		list-style: none;
		padding: 0;
	}
	.timeline li {
		border-left: 2px solid #087f8c;
		padding: 10px 0 14px 14px;
		margin-left: 5px;
		border-radius: 0 10px 10px 0;
	}
	.time-head {
		display: flex;
		align-items: center;
		gap: 9px;
		flex-wrap: wrap;
	}
	.time-head > span,
	.time-head > small {
		font-size: 11px;
		opacity: 0.75;
	}
	.time-head button {
		text-align: left;
	}
	.timeline summary {
		min-height: 44px;
		cursor: pointer;
		display: flex;
		align-items: center;
		font-size: 12px;
	}
	.timeline p {
		font-size: 13px;
		overflow-wrap: anywhere;
	}
	.controls {
		display: grid;
		gap: 15px;
	}
	.control {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		min-height: 44px;
		flex-wrap: wrap;
	}
	.control > span {
		font-size: 13px;
		min-width: 70px;
	}
	.control input[type='checkbox'] {
		width: 44px;
		height: 24px;
		accent-color: #087f8c;
	}
	.numeric {
		display: flex;
		gap: 12px;
		align-items: center;
		flex: 1;
		min-width: 150px;
		max-width: 430px;
	}
	.numeric input[type='range'] {
		min-height: 44px;
		width: 100%;
		min-width: 40px;
		accent-color: #087f8c;
	}
	.numeric input[type='number'] {
		width: 105px;
		max-width: 100%;
	}
	.control select {
		max-width: 70%;
	}
	.outputs {
		display: flex;
		gap: 12px;
		flex-wrap: wrap;
		margin: 18px 0;
	}
	.outputs > div {
		min-width: 110px;
		flex: 1;
		border-radius: 12px;
		background: color-mix(in srgb, #087f8c 8%, transparent);
		padding: 12px;
	}
	.outputs small {
		display: block;
		font-size: 11px;
		opacity: 0.7;
	}
	.outputs strong {
		display: block;
		font-size: 20px;
		font-variant-numeric: tabular-nums;
		margin-top: 4px;
	}
	.diagram {
		width: 100%;
		max-height: 320px;
	}
	.diagram line {
		stroke: currentColor;
		stroke-width: 1.5;
		opacity: 0.3;
	}
	.diagram rect {
		fill: var(--background, #fff);
		stroke: #087f8c;
		stroke-width: 1.5;
	}
	.diagram g {
		cursor: pointer;
	}
	.diagram g.selected rect {
		fill: color-mix(in srgb, #087f8c 18%, var(--background, #fff));
		stroke-width: 3;
	}
	.diagram text {
		fill: currentColor;
		font-size: 12px;
	}
	.node-buttons {
		display: flex;
		gap: 7px;
		flex-wrap: wrap;
	}
	.node-detail {
		padding: 12px;
		margin-top: 12px;
		border-radius: 10px;
		background: color-mix(in srgb, #087f8c 8%, transparent);
	}
	.node-detail p {
		margin: 5px 0 0;
		overflow-wrap: anywhere;
	}
	@container (max-width:360px) {
		.ui-card header {
			padding: 14px;
		}
		.body {
			padding: 0 12px 12px;
		}
		footer {
			padding: 10px 12px;
		}
		footer > div {
			width: 100%;
		}
		footer button {
			flex: 1;
		}
		.mark {
			display: none;
		}
		.numeric {
			flex-basis: 100%;
			max-width: none;
		}
	}
</style>

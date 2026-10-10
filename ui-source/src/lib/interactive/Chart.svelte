<script lang="ts">
	let {
		data,
		viewState = {},
		onVisible
	}: {
		data: any;
		viewState?: Record<string, any>;
		onVisible?: (key: string, value: boolean) => void;
	} = $props();
	const colors = [
		'#087f8c',
		'#3867d6',
		'#a15b00',
		'#8753bf',
		'#bd3c70',
		'#328348',
		'#a44428',
		'#526578'
	];
	let tip = $state('');
	let localHidden = $state<string[]>([]);
	const shown = $derived(
		(data?.series ?? []).filter(
			(s: any) => viewState['visible:' + s.id] !== false && !localHidden.includes(s.id)
		)
	);
	const values = $derived(shown.flatMap((s: any) => s.values as number[]));
	const lo = $derived(Math.min(0, ...values));
	const hi = $derived(Math.max(1, ...values));
	const x = (i: number) => 48 + (i * 500) / Math.max(1, data.labels.length - 1);
	const y = (v: number) => 200 - ((v - lo) * 170) / Math.max(1e-9, hi - lo);
	const path = (s: any) =>
		s.values.map((v: number, i: number) => (i ? 'L' : 'M') + x(i) + ',' + y(v)).join(' ');
	const format = (v: number) =>
		new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(v);
	function describe(s: any, i: number) {
		tip = `${s.label} · ${data.labels[i]}: ${format(s.values[i])}`;
	}
	function toggle(s: any) {
		const visible = viewState['visible:' + s.id] !== false && !localHidden.includes(s.id);
		if (onVisible) onVisible('visible:' + s.id, !visible);
		else localHidden = visible ? [...localHidden, s.id] : localHidden.filter((id) => id !== s.id);
	}
	function arcs(s: any) {
		let a = -Math.PI / 2;
		const total = s.values.reduce((n: number, v: number) => n + v, 0);
		return s.values.map((v: number, i: number) => {
			const start = a,
				end = a + (total ? (v / total) * 2 * Math.PI : 0);
			a = end;
			const r = 84,
				cx = 300,
				cy = 110;
			return {
				path:
					v === total && v > 0
						? `M ${cx} ${cy - r} A ${r} ${r} 0 1 1 ${cx - 0.001} ${cy - r} Z`
						: `M ${cx} ${cy} L ${cx + r * Math.cos(start)} ${cy + r * Math.sin(start)} A ${r} ${r} 0 ${end - start > Math.PI ? 1 : 0} 1 ${cx + r * Math.cos(end)} ${cy + r * Math.sin(end)} Z`,
				i,
				value: v
			};
		});
	}
</script>

<div class="chart">
	<div class="legend" aria-label="筛选数据系列">
		{#each data.series as s, i (s.id)}<button
				aria-pressed={viewState['visible:' + s.id] !== false && !localHidden.includes(s.id)}
				onclick={() => toggle(s)}
				style={`--series:${colors[i % colors.length]}`}
				type="button"><span class="dot"></span>{s.label}</button
			>{/each}
	</div>
	<svg aria-label={`${data.kind} 图表，完整数值见下方数据表`} role="img" viewBox="0 0 600 238">
		<title>交互数据图</title><desc>可筛选系列，选择数据点查看数值；也可使用文字数据表。</desc>
		{#if data.kind === 'pie' && shown.length}
			{#each arcs(shown[0]) as a}<path
					d={a.path}
					fill={colors[a.i % colors.length]}
					role="button"
					tabindex="0"
					aria-label={`${data.labels[a.i]}: ${format(a.value)}`}
					onkeydown={(e) => {
						if (e.key === 'Enter' || e.key === ' ') {
							e.preventDefault();
							describe(shown[0], a.i);
						}
					}}
					onclick={() => describe(shown[0], a.i)}
					><title>{data.labels[a.i]}: {format(a.value)}</title></path
				>{/each}
		{:else}
			<line class="axis" x1="48" x2="548" y1={y(0)} y2={y(0)} /><text class="tick" x="8" y="35"
				>{format(hi)}</text
			><text class="tick" x="8" y="202">{format(lo)}</text>
			{#each shown as s, si (s.id)}
				{#if data.kind === 'line'}<path
						d={path(s)}
						fill="none"
						stroke={colors[data.series.indexOf(s) % colors.length]}
						stroke-width="2.5"
					/>{/if}
				{#each s.values as v, i}
					{#if data.kind === 'bar'}<rect
							fill={colors[data.series.indexOf(s) % colors.length]}
							height={Math.max(1, Math.abs(y(v) - y(0)))}
							width={Math.max(2, 420 / Math.max(1, data.labels.length) / Math.max(1, shown.length))}
							x={48 +
								((i + 0.15) * 500) / data.labels.length +
								(si * 420) / data.labels.length / shown.length}
							y={Math.min(y(0), y(v))}
							role="button"
							tabindex="0"
							aria-label={`${s.label} ${data.labels[i]}: ${format(v)}`}
							onkeydown={(e) => {
								if (e.key === 'Enter' || e.key === ' ') {
									e.preventDefault();
									describe(s, i);
								}
							}}
							onclick={() => describe(s, i)}
							><title>{s.label} · {data.labels[i]}: {format(v)}</title></rect
						>
					{:else}<circle
							cx={x(i)}
							cy={y(v)}
							fill={colors[data.series.indexOf(s) % colors.length]}
							r="4"
							role="button"
							tabindex="0"
							aria-label={`${s.label} ${data.labels[i]}: ${format(v)}`}
							onkeydown={(e) => {
								if (e.key === 'Enter' || e.key === ' ') {
									e.preventDefault();
									describe(s, i);
								}
							}}
							onclick={() => describe(s, i)}
							><title>{s.label} · {data.labels[i]}: {format(v)}</title></circle
						>{/if}
				{/each}
			{/each}
			{#each data.labels as label, i}{#if i === 0 || i === data.labels.length - 1 || data.labels.length <= 8}<text
						class="tick"
						text-anchor="middle"
						x={data.kind === 'bar' ? 48 + ((i + 0.5) * 500) / data.labels.length : x(i)}
						y="229">{label.length > 12 ? label.slice(0, 12) + '…' : label}</text
					>{/if}{/each}
		{/if}
	</svg>
	{#if data.kind === 'pie' && shown.length}<div class="pie-labels">
			{#each data.labels as label, i}<span
					><i style={`background:${colors[i % colors.length]}`}></i>{label}: {format(
						shown[0].values[i]
					)}</span
				>{/each}
		</div>{/if}
	<p aria-live="polite" class="tip">{tip || '点击图形或数据表查看数值'}</p>
	<details>
		<summary>文字数据表</summary>
		<div class="table-wrap" role="group" aria-label="图表完整数据">
			<table>
				<thead
					><tr
						><th scope="col">项目</th>{#each shown as s (s.id)}<th scope="col">{s.label}</th
							>{/each}</tr
					></thead
				><tbody
					>{#each data.labels as label, i}<tr
							><th scope="row">{label}</th>{#each shown as s (s.id)}<td
									><button onclick={() => describe(s, i)} type="button"
										>{format(s.values[i])}</button
									></td
								>{/each}</tr
						>{/each}</tbody
				>
			</table>
		</div>
	</details>
</div>

<style>
	.chart {
		min-width: 0;
		max-width: 100%;
	}
	.legend,
	.pie-labels {
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
	}
	.legend button {
		min-height: 44px;
		padding: 8px 12px;
		border: 1px solid var(--border, #ccd5d6);
		border-radius: 10px;
		background: transparent;
		font: inherit;
		color: inherit;
		display: flex;
		align-items: center;
		gap: 7px;
	}
	.legend button[aria-pressed='false'] {
		opacity: 0.55;
		text-decoration: line-through;
	}
	.dot {
		width: 9px;
		height: 9px;
		background: var(--series);
		border-radius: 50%;
	}
	svg {
		width: 100%;
		max-height: 260px;
		overflow: visible;
	}
	.axis {
		stroke: currentColor;
		opacity: 0.25;
	}
	.tick {
		fill: currentColor;
		font-size: 12px;
	}
	.tip {
		font-size: 12px;
		min-height: 20px;
		opacity: 0.75;
	}
	.pie-labels span {
		font-size: 12px;
		display: flex;
		align-items: center;
		gap: 5px;
	}
	.pie-labels i {
		width: 8px;
		height: 8px;
		border-radius: 50%;
	}
	summary {
		cursor: pointer;
		min-height: 44px;
		display: flex;
		align-items: center;
		font-size: 13px;
	}
	.table-wrap {
		overflow: auto;
		max-height: 300px;
	}
	table {
		border-collapse: collapse;
		width: 100%;
		font-size: 13px;
	}
	th,
	td {
		border-bottom: 1px solid var(--border, #ccd5d6);
		padding: 7px;
		text-align: left;
		white-space: nowrap;
	}
	td button {
		min-height: 44px;
		min-width: 44px;
		border: 0;
		background: transparent;
		color: inherit;
		font: inherit;
		padding: 4px;
	}
	:focus-visible {
		outline: 2px solid #078596;
		outline-offset: 3px;
	}
</style>

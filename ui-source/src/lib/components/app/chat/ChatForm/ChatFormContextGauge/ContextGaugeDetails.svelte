<script lang="ts">
	import ContextGaugeDetailRow from './ContextGaugeDetailRow.svelte';
	import { gaugePopup } from './gauge-popup.svelte';
	import { ChevronDown } from '@lucide/svelte';
	import * as Collapsible from '$lib/components/ui/collapsible';
	import { STATS_UNITS } from '$lib/constants';
	import { conversationsStore } from '$lib/stores';
	import { speculativeStats, kvCacheStats } from '$lib/utils/speculative-stats';
	const speculation = $derived(speculativeStats(conversationsStore.activeMessages));
	const kvStats = $derived(kvCacheStats(conversationsStore.activeMessages));

	interface Props {
		currentRead: number;
		currentFresh: number;
		currentCache: number;
		currentOutput: number;
		kvTotal: number;
		cumulativeRead: number;
		cumulativeOutput: number;
		cumulativeCacheTotal: number;
		averageTokensPerSecond: number | null;
		transientDetails: string[];
	}

	let {
		averageTokensPerSecond,
		cumulativeCacheTotal,
		cumulativeOutput,
		cumulativeRead,
		currentCache,
		currentFresh,
		currentOutput,
		currentRead,
		kvTotal,
		transientDetails
	}: Props = $props();

	const hasCumulative = $derived(cumulativeRead > 0 || cumulativeOutput > 0);
	const hasCurrent = $derived(currentRead > 0 || currentOutput > 0);
	const hitRate=(cached:number,total:number)=>total>0?`${(Math.min(cached,total)/total*100).toFixed(1)}%`:'—';
</script>

<Collapsible.Root bind:open={gaugePopup.detailsOpen} class="mt-3 border-t border-border/50 pt-4">
	<Collapsible.Trigger
		class="flex w-full cursor-pointer items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
	>
		<span>Token usage details</span>

		<ChevronDown
			class={'ml-auto h-3 w-3 transition-transform' + (gaugePopup.detailsOpen ? ' rotate-180' : '')}
		/>
	</Collapsible.Trigger>

	<Collapsible.Content class="flex flex-col gap-4 text-xs pt-4">
		<section aria-label="投机解码统计">
			<h3 class="text-[11px] font-medium text-muted-foreground/70 mb-2">投机解码 · 当前会话</h3>
			<div class="flex flex-col gap-2" aria-live="polite">
				<ContextGaugeDetailRow label="接受率" subtitle={`${speculation.accepted.toLocaleString()} / ${speculation.proposed.toLocaleString()} 个提议 token`} value={speculation.acceptanceRate===null?(speculation.seen?(speculation.enabled?'未发起投机':'已关闭'):'暂无统计'):`${(speculation.acceptanceRate*100).toFixed(1)}%`} />
				<ContextGaugeDetailRow label="平均连续接受长度" subtitle="每个验证批次的接受前缀，包含零接受批次" value={speculation.averageAcceptedLength===null?'—':`${speculation.averageAcceptedLength.toFixed(2)} tok`} />
				<ContextGaugeDetailRow label="累计接受 token" subtitle={`${speculation.batches.toLocaleString()} 个验证批次 · 当前分支`} value={`${speculation.accepted.toLocaleString()} tok`} />
				{#if speculation.throttled}<p class="text-muted-foreground">部分生成因接受率低而自动暂停投机。</p>{/if}
			</div>
		</section>
		{#if hasCumulative}
			<div>
				<h3 class="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/70 mb-2">
					Across all turns
				</h3>

				<div class="flex flex-col gap-2">
					<ContextGaugeDetailRow label="KV cache hit rate" subtitle={`${kvStats.cached.toLocaleString()} / ${kvStats.total.toLocaleString()} prompt tok · 累计`} value={hitRate(kvStats.cached,kvStats.total)} />
					{#if cumulativeRead > 0}
						<ContextGaugeDetailRow
							label="Prompt tokens evaluated"
							subtitle={cumulativeCacheTotal > 0
								? `${cumulativeCacheTotal.toLocaleString()} reused from KV cache`
								: undefined}
							value={`${cumulativeRead.toLocaleString()} tok`}
						/>
					{/if}

					{#if cumulativeOutput > 0}
						<ContextGaugeDetailRow
							label="Tokens generated"
							value={`${cumulativeOutput.toLocaleString()} tok`}
						/>
					{/if}
				</div>
			</div>
		{/if}

		{#if hasCurrent}
			<div>
				<h3 class="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/70 mb-2">
					This turn · KV cache
				</h3>

				<div class="flex flex-col gap-2">
					<ContextGaugeDetailRow label="KV cache hit rate" subtitle={`${kvStats.currentCached.toLocaleString()} / ${kvStats.currentTotal.toLocaleString()} prompt tok · 本次生成`} value={hitRate(kvStats.currentCached,kvStats.currentTotal)} />
					{#if currentRead > 0}
						<ContextGaugeDetailRow
							label="Prompt"
							subtitle={currentCache > 0
								? `${currentFresh.toLocaleString()} fresh + ${currentCache.toLocaleString()} cached`
								: undefined}
							value={`${currentRead.toLocaleString()} tok`}
						/>
					{/if}

					{#if currentOutput > 0}
						<ContextGaugeDetailRow
							label="Generated"
							value={`${currentOutput.toLocaleString()} tok`}
						/>
					{/if}

					<div class="pt-1 mt-0.5 border-t border-border/30">
						<div class="flex justify-between">
							<span class="text-muted-foreground">KV cache total</span>

							<span class="font-mono font-medium">{kvTotal.toLocaleString()} tok</span>
						</div>
					</div>
				</div>
			</div>
		{/if}

		{#if averageTokensPerSecond !== null}
			<div class="pt-1.5 mt-1 border-t border-border/30">
				<ContextGaugeDetailRow
					label="Avg speed"
					value={`${averageTokensPerSecond.toFixed(1)}${STATS_UNITS.TOKENS_PER_SECOND}`}
				/>
			</div>
		{/if}

		{#each transientDetails as detail (detail)}
			<div class="font-mono text-muted-foreground">{detail}</div>
		{/each}
	</Collapsible.Content>
</Collapsible.Root>

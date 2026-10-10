<script lang="ts">
	import { X } from '@lucide/svelte';
	import { ActionIcon } from '$lib/components/app';

	interface Props {
		class?: string;
		isLoading?: boolean;
		loadError?: string;
		height?: string;
		id: string;
		imageClass?: string;
		onclick?: (event?: MouseEvent) => void;
		onRemove?: (id: string) => void;
		name: string;
		preview: string;
		readonly?: boolean;
		width?: string;
	}

	let {
		class: className = '',
		isLoading = false,
		loadError,
		height = 'h-16',
		id,
		imageClass = '',
		name,
		onclick,
		onRemove,
		preview,
		readonly = false,
		width = 'w-auto'
	}: Props = $props();
</script>

{#snippet image()}
	<img alt={name} class="{height} {width} cursor-pointer object-cover {imageClass}" src={preview} />
{/snippet}

<div
	class="group relative min-w-0 overflow-hidden rounded-lg bg-muted shadow-lg dark:border dark:border-muted {className}"
>
	{#if onclick}
		<button
			aria-label="Preview {name}"
			class="block h-full w-full rounded-lg focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:outline-none"
			{onclick}
			type="button"
		>
			{@render image()}
		</button>
	{:else}
		{@render image()}
	{/if}

	{#if isLoading || loadError}<span role={loadError?'alert':'status'} class="absolute bottom-0 left-0 max-w-full bg-black/75 px-2 py-1 text-xs text-white">{loadError || '正在处理…'}</span>{/if}
	{#if !readonly}
		<div
			class="absolute top-1 right-1 flex items-center justify-center opacity-100"
		>
			<ActionIcon
				class="text-white"
				icon={X}
				onclick={() => onRemove?.(id)}
				stopPropagationOnClick
				tooltip="Remove"
			/>
		</div>
	{/if}
</div>

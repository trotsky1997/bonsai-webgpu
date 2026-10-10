<script lang="ts">
	import InteractiveCard from '$lib/interactive/InteractiveCard.svelte';
	import ChatMessageToolCallBlockDefault from './ChatMessageToolCallBlockDefault.svelte';
	import ChatMessageToolCallBlockEditFile from './ChatMessageToolCallBlockEditFile.svelte';
	import ChatMessageToolCallBlockExecShellCommand from './ChatMessageToolCallBlockExecShellCommand.svelte';
	import ChatMessageToolCallBlockFileGlobSearch from './ChatMessageToolCallBlockFileGlobSearch.svelte';
	import ChatMessageToolCallBlockGetDatetime from './ChatMessageToolCallBlockGetDatetime.svelte';
	import ChatMessageToolCallBlockGetInfo from './ChatMessageToolCallBlockGetInfo.svelte';
	import ChatMessageToolCallBlockGrepSearch from './ChatMessageToolCallBlockGrepSearch.svelte';
	import ChatMessageToolCallBlockReadFile from './ChatMessageToolCallBlockReadFile.svelte';
	import ChatMessageToolCallBlockReadMedia from './ChatMessageToolCallBlockReadMedia.svelte';
	import ChatMessageToolCallBlockRunJavascript from './ChatMessageToolCallBlockRunJavascript.svelte';
	import ChatMessageToolCallBlockSearchResults from './ChatMessageToolCallBlockSearchResults.svelte';
	import ChatMessageToolCallBlockWriteFile from './ChatMessageToolCallBlockWriteFile.svelte';
	import { BuiltInTool } from '$lib/enums';
	import type { AgenticSection, DatabaseMessageExtra } from '$lib/types';
	import {
		extractSearchQuery,
		extractSearchResults,
		isWebSearchToolName,
		looksLikeSearchResult
	} from '$lib/utils';

	interface Props {
		section: AgenticSection;
		conversationId?: string;
		attachments?: DatabaseMessageExtra[];
		open: boolean;
		isStreaming: boolean;
		isExecuting?: boolean;
		onToggle?: () => void;
	}

	let {
		conversationId = '',
		attachments,
		isExecuting,
		isStreaming,
		onToggle,
		open,
		section
	}: Props = $props();

	// Runs for every tool block on mount, before the body renders: the cheap
	// content prefilter and the tool-name allow-list come first so blobs from
	// exec/file tools are never line-split or JSON-parsed here
	const isSearchCall = $derived.by(() => {
		if (looksLikeSearchResult(section.toolResult)) {
			return extractSearchResults(section.toolResult).length > 0;
		}

		return isWebSearchToolName(section.toolName) && extractSearchQuery(section.toolArgs).length > 0;
	});
	const uiArgs = $derived.by(() => {
		try {
			return JSON.parse(section.toolArgs ?? '{}');
		} catch {
			return null;
		}
	});
</script>

{#if section.toolName === 'present_ui' && uiArgs && section.toolCallId && conversationId}
	{#key section.toolCallId}<InteractiveCard
			args={uiArgs}
			callId={section.toolCallId}
			{conversationId}
		/>{/key}
{:else if section.toolName === 'update_ui_state'}
	<details class="my-2 rounded-xl border p-3 text-sm">
		<summary>更新交互组件状态</summary>
		<pre class="overflow-auto">{section.toolResult ?? '正在更新…'}</pre>
	</details>
{:else if isSearchCall}
	<ChatMessageToolCallBlockSearchResults {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.BROWSER_GET_DATETIME}
	<ChatMessageToolCallBlockGetDatetime {isStreaming} {section} />
{:else if section.toolName === BuiltInTool.SERVER_GET_INFO}
	<ChatMessageToolCallBlockGetInfo {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.SERVER_READ_FILE}
	<ChatMessageToolCallBlockReadFile {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.BROWSER_READ_MEDIA}
	<ChatMessageToolCallBlockReadMedia {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.SERVER_EDIT_FILE}
	<ChatMessageToolCallBlockEditFile {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.SERVER_WRITE_FILE}
	<ChatMessageToolCallBlockWriteFile {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.SERVER_EXEC_SHELL_COMMAND}
	<ChatMessageToolCallBlockExecShellCommand
		{attachments}
		{isExecuting}
		{isStreaming}
		{onToggle}
		{open}
		{section}
	/>
{:else if section.toolName === BuiltInTool.SERVER_FILE_GLOB_SEARCH}
	<ChatMessageToolCallBlockFileGlobSearch {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.SERVER_GREP_SEARCH}
	<ChatMessageToolCallBlockGrepSearch {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.BROWSER_RUN_JAVASCRIPT}
	<ChatMessageToolCallBlockRunJavascript {isStreaming} {onToggle} {open} {section} />
{:else}
	<ChatMessageToolCallBlockDefault {attachments} {isStreaming} {onToggle} {open} {section} />
{/if}

export type UiKind = 'comparison' | 'checklist' | 'timeline' | 'chart' | 'parameters' | 'diagram';
export type UiState = Record<string, string | number | boolean>;
export interface UiDefinition {
	version: 1;
	id: string;
	type: UiKind;
	title: string;
	summary: string;
	data: Record<string, any>;
}
export type UiContent =
	| { kind: 'text'; text: string }
	| { kind: 'component'; id: string; version: 1; summary: string; callId: string; offset: number };
export interface UiRecord {
	key: string;
	conversationId: string;
	calls: string[];
	definition: UiDefinition;
	version: number;
	state: UiState;
	history: { version: number; state: UiState }[];
	operations: string[];
	updatedAt: number;
}
export interface UiPatchResult {
	status: 'ok' | 'conflict';
	record: UiRecord;
	fields?: string[];
}
export interface UiCalculation {
	outputs: { id: string; label: string; unit?: string; value: any }[];
	chart: any;
}

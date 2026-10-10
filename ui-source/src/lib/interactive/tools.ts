import type { OpenAIToolDefinition } from '$lib/types';
export const UI_TOOLS = ['present_ui', 'update_ui_state'];
const text = { type: 'string' };
const number = { type: 'number' };
const rows = (properties: Record<string, unknown>, required: string[]) => ({
	type: 'array',
	maxItems: 16,
	items: { type: 'object', properties, required }
});
const named = { id: { type: 'string', description: 'Short ASCII id, no spaces' }, label: text };
export function uiTools(): OpenAIToolDefinition[] {
	return [
		{
			type: 'function',
			function: {
				name: 'present_ui',
				description:
					'Display an interactive card in the chat. Always include type, title, summary and data. Choose only the data fields for your type: checklist/timeline use items; comparison uses columns+rows; chart uses kind+labels+series; parameters uses controls+outputs; diagram uses nodes+edges. Provide 2-6 real items. After success respond briefly without repeating the card.',
				parameters: {
					type: 'object',
					properties: {
						type: {
							type: 'string',
							enum: ['comparison', 'checklist', 'timeline', 'chart', 'parameters', 'diagram']
						},
						title: { type: 'string', description: 'Required card heading' },
						summary: { type: 'string', description: 'Required text summary' },
						id: { type: 'string', description: 'Optional ASCII id; usually omit' },
						data: {
							type: 'object',
							description: 'REQUIRED. Fill the matching fields below with actual data.',
							properties: {
								items: rows(
									{
										...named,
										group: text,
										description: text,
										date: text,
										checked: { type: 'boolean' }
									},
									['id', 'label']
								),
								columns: rows({ key: text, label: text }, ['key', 'label']),
								rows: rows(
									{
										...named,
										values: { type: 'object', description: 'Column key to string or number value' }
									},
									['id', 'label', 'values']
								),
								kind: { type: 'string', enum: ['line', 'bar', 'pie'] },
								labels: { type: 'array', items: text },
								series: rows({ ...named, values: { type: 'array', items: number } }, [
									'id',
									'label',
									'values'
								]),
								controls: rows(
									{
										...named,
										kind: { type: 'string', enum: ['number', 'slider', 'select', 'switch'] },
										min: number,
										max: number,
										step: number,
										value: { type: ['number', 'boolean', 'string'] },
										options: rows({ value: text, label: text }, ['value', 'label'])
									},
									['id', 'label', 'kind']
								),
								outputs: rows(
									{
										...named,
										unit: text,
										expr: {
											description:
												'Calculation AST: a number, {param:"control_id"}, or {op:"add|sub|mul|div|if|range|map|sum|avg",args:[expressions]}. Example {op:"mul",args:[{param:"price"},{param:"count"}]}. map binds {var:"x"}.',
											type: ['number', 'object', 'array']
										}
									},
									['id', 'label', 'expr']
								),
								chart: {
									type: 'object',
									description:
										'Optional calculated chart {kind:"line",series:[{id,label,expr}]} where expr produces an array, e.g. {op:"map",args:[{op:"range",args:[1,12,1]},{op:"mul",args:[{var:"x"},{param:"price"}]}]}'
								},
								nodes: rows(
									{
										...named,
										x: { ...number, minimum: 0, maximum: 100 },
										y: { ...number, minimum: 0, maximum: 100 },
										description: text
									},
									['id', 'label', 'x', 'y']
								),
								edges: rows({ from: text, to: text, label: text }, ['from', 'to'])
							}
						}
					},
					required: ['type', 'title', 'summary', 'data']
				}
			}
		},
		{
			type: 'function',
			function: {
				name: 'update_ui_state',
				description:
					'Update an existing card using its id and version. patch flat keys: selected; checked:item-id (boolean); visible:series-id (boolean); param:control-id. Same-field concurrent changes return a conflict.',
				parameters: {
					type: 'object',
					properties: { id: text, base_version: { type: 'integer' }, patch: { type: 'object' } },
					required: ['id', 'base_version', 'patch']
				}
			}
		}
	];
}

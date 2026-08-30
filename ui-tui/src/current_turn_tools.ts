export type ToolActivity = {
	id: string;
	toolName: string;
	summary: string;
	status: 'running' | 'success' | 'error' | 'called';
};

type ToolStart = {type: 'tool.start'; id: string; toolName: string; summary: string};
type ToolEnd = {type: 'tool.end'; id: string; status: 'success' | 'error'};

export function beginCurrentTurn(_previous: ToolActivity[] = []): ToolActivity[] {
	return [];
}

export function applyCurrentTurnToolEvent(current: ToolActivity[], event: ToolStart | ToolEnd): ToolActivity[] {
	if (event.type === 'tool.start') {
		return [...current, {id: event.id, toolName: event.toolName, summary: event.summary, status: 'running'}];
	}
	return current.map(tool => tool.id === event.id ? {...tool, status: event.status} : tool);
}

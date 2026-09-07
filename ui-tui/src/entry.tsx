import React, {useEffect, useMemo, useState} from 'react';
import {Box, render, Text, useApp, useInput} from 'ink';

import {GatewayClient, type GatewayEvent, type RuntimeStatus, type SessionSummary, type TranscriptItem} from './gatewayClient.js';
import {applyCurrentTurnToolEvent, beginCurrentTurn, type ToolActivity} from './current_turn_tools.js';
import {Markdown} from './markdown.js';
import {TUI_RENDER_OPTIONS} from './tui_render_options.js';

type Approval = {requestId: string; toolName: string; summary: string; policyReason: string; command?: string; path?: string; cwd?: string};

function value(payload: Record<string, unknown>, key: string): string {
	return typeof payload[key] === 'string' ? payload[key] : '';
}

function App(): React.ReactNode {
	const {exit} = useApp();
	const client = useMemo(() => new GatewayClient(), []);
	const [status, setStatus] = useState<RuntimeStatus | null>(null);
	const [transcript, setTranscript] = useState<TranscriptItem[]>([]);
	const [streaming, setStreaming] = useState('');
	const [tools, setTools] = useState<ToolActivity[]>([]);
	const [approval, setApproval] = useState<Approval | null>(null);
	const [sessions, setSessions] = useState<SessionSummary[] | null>(null);
	const [notice, setNotice] = useState('');
	const [busy, setBusy] = useState(false);
	const [cancelling, setCancelling] = useState(false);

	useEffect(() => {
		const unsubscribe = client.onEvent(event => handleEvent(event));
		void client.request<RuntimeStatus>('runtime.status').then(setStatus).catch(error => setNotice(error.message));
		return unsubscribe;
		// handleEvent intentionally reads current setters only.
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [client]);

	function handleEvent(event: GatewayEvent): void {
		const payload = event.payload;
		switch (event.type) {
			case 'assistant.start':
				setStreaming('');
				break;
			case 'assistant.delta':
				setStreaming(current => current + value(payload, 'text'));
				break;
			case 'assistant.end': {
				const completed = value(payload, 'text');
				setStreaming(current => {
					const text = current || completed;
					if (text) setTranscript(messages => [...messages, {role: 'assistant', text}]);
					return '';
				});
				break;
			}
			case 'tool.start': {
				const id = value(payload, 'tool_call_id');
				setTools(current => applyCurrentTurnToolEvent(current, {
					type: 'tool.start', id, toolName: value(payload, 'tool_name'), summary: value(payload, 'summary'),
				}));
				break;
			}
			case 'tool.end': {
				const id = value(payload, 'tool_call_id');
				const nextStatus = value(payload, 'status') === 'error' ? 'error' : 'success';
				setTools(current => applyCurrentTurnToolEvent(current, {type: 'tool.end', id, status: nextStatus}));
				break;
			}
			case 'approval.request':
				setApproval({
					requestId: value(payload, 'request_id'),
					toolName: value(payload, 'tool_name'),
					summary: value(payload, 'summary'),
					policyReason: value(payload, 'policy_reason'),
					command: value(payload, 'command') || undefined,
					path: value(payload, 'path') || undefined,
					cwd: value(payload, 'cwd') || undefined,
				});
				break;
			case 'run.finished':
				setBusy(false);
				setCancelling(false);
				break;
			case 'run.cancelled':
				setBusy(false);
				setCancelling(false);
				setApproval(null);
				setStreaming('');
				setTools(beginCurrentTurn());
				setNotice(value(payload, 'message') || 'Turn cancelled by user.');
				break;
			case 'error':
				setBusy(false);
				setCancelling(false);
				setNotice(`${value(payload, 'error_type') || 'Error'}: ${value(payload, 'message')}`);
				break;
		}
	}

	async function submit(text: string): Promise<void> {
		if (busy || !text.trim()) return;
		try {
			setTools(beginCurrentTurn());
			setStreaming('');
			await client.request<{accepted: boolean}>('prompt.submit', {text});
			setTranscript(current => [...current, {role: 'user', text}]);
			setBusy(true);
			setNotice('');
		} catch (error) {
			setNotice(error instanceof Error ? error.message : String(error));
		}
	}

	async function cancelCurrentTurn(): Promise<void> {
		if (!busy || cancelling) return;
		setCancelling(true);
		try {
			await client.request<{cancelled: boolean}>('prompt.cancel');
		} catch (error) {
			setCancelling(false);
			setNotice(error instanceof Error ? error.message : String(error));
		}
	}

	async function chooseApproval(decision: 'allow' | 'deny'): Promise<void> {
		if (!approval) return;
		try {
			await client.request('approval.respond', {request_id: approval.requestId, decision});
		} catch (error) {
			setNotice(error instanceof Error ? error.message : String(error));
		} finally {
			setApproval(null);
		}
	}

	async function openSessions(): Promise<void> {
		if (busy) return;
		try {
			setSessions(await client.request<SessionSummary[]>('session.list'));
		} catch (error) {
			setNotice(error instanceof Error ? error.message : String(error));
		}
	}

	async function resume(sessionId: string): Promise<void> {
		try {
			const result = await client.request<{status: RuntimeStatus; transcript: TranscriptItem[]}>('session.resume', {session_id: sessionId});
			setStatus(result.status);
			setTranscript(result.transcript);
			setTools(beginCurrentTurn());
			setSessions(null);
		} catch (error) {
			setNotice(error instanceof Error ? error.message : String(error));
		}
	}

	async function newSession(): Promise<void> {
		if (busy) return;
		try {
			setStatus(await client.request<RuntimeStatus>('session.new'));
			setTranscript([]);
			setTools(beginCurrentTurn());
			setStreaming('');
			setNotice('New session created.');
		} catch (error) {
			setNotice(error instanceof Error ? error.message : String(error));
		}
	}

	async function close(): Promise<void> {
		if (busy) {
			setNotice('A run is active; wait for it to finish before exiting.');
			return;
		}
		await client.close();
		exit();
	}

	useInput((input, key) => {
		if (busy && key.ctrl && input === 'c') {
			void cancelCurrentTurn();
			return;
		}
		if (approval || sessions) return;
		if (busy && key.escape) {
			void cancelCurrentTurn();
			return;
		}
		if (key.ctrl && input === 'n') void newSession();
		else if (key.ctrl && input === 'r') void openSessions();
		else if (key.ctrl && input === 'c') void close();
	});

	return <Box flexDirection="column" paddingX={1}>
		<StatusBar status={status}/>
		<Box flexDirection="column" marginY={1}>
			{transcript.filter(item => item.role !== 'tool').map((item, index) => <Message key={index} item={item}/>)}
			{tools.map(tool => <ToolLine key={tool.id} tool={tool}/>) }
			{streaming ? <Box flexDirection="column"><Text bold color="green">Rova</Text><Markdown text={streaming}/></Box> : null}
			{busy ? <Text color="yellow">{cancelling ? 'cancelling…' : 'working…'}</Text> : null}
			{notice ? <Text color="yellow">{notice}</Text> : null}
		</Box>
		<Composer busy={busy} onSubmit={submit}/>
		<Text dimColor>{busy ? 'Esc / Ctrl+C cancel current turn' : 'Ctrl+N new session · Ctrl+R sessions · Ctrl+C exit when idle'}</Text>
		{approval ? <ApprovalModal approval={approval} onDecision={chooseApproval} onCancel={cancelCurrentTurn}/> : null}
		{sessions ? <SessionPicker sessions={sessions} onResume={resume} onClose={() => setSessions(null)}/> : null}
	</Box>;
}

function StatusBar({status}: {status: RuntimeStatus | null}): React.ReactNode {
	const items = ['Rova'];
	if (status) {
		items.push(`model: ${status.model}`, `session: ${status.session_id?.slice(0, 8) ?? 'none'}`);
		if (status.workspace) items.push(`workspace: ${status.workspace}`);
		if (status.recovery.recovered_count) {
			items.push(`recovery:${status.recovery.recovered_count}`);
			if (status.recovery.side_effects_unknown_count) items.push(`side-effects?:${status.recovery.side_effects_unknown_count}`);
		}
		if (status.terminal_backend) items.push(`terminal: ${status.terminal_backend.kind} (${status.terminal_backend.cwd})`);
		if (status.web_enabled) items.push('web:on');
		if (status.skill_count) items.push(`skills:${status.skill_count}`);
		if (status.mcp) {
			const states = Object.values(status.mcp.server_states);
			items.push(`mcp:${states.filter(state => state === 'ready').length}/${states.length}`);
		}
	}
	return <Box borderStyle="round" paddingX={1}><Text bold>{items.join(' | ')}</Text></Box>;
}

function Message({item}: {item: TranscriptItem}): React.ReactNode {
	return <Box flexDirection="column" marginBottom={1}>
		<Text bold color={item.role === 'user' ? 'blue' : 'green'}>{item.role === 'user' ? 'You' : 'Rova'}</Text>
		<Markdown text={item.text ?? ''}/>
	</Box>;
}

function ToolLine({tool}: {tool: ToolActivity}): React.ReactNode {
	const marker = tool.status === 'running' ? '…' : tool.status === 'error' ? '✗' : '✓';
	return <Box marginLeft={1}><Text color={tool.status === 'error' ? 'red' : tool.status === 'running' ? 'yellow' : 'gray'}>▸ {tool.toolName}  {tool.summary}  {marker}</Text></Box>;
}

function Composer({busy, onSubmit}: {busy: boolean; onSubmit: (value: string) => Promise<void>}): React.ReactNode {
	const [value, setValue] = useState('');
	useInput((input, key) => {
		if (busy || key.ctrl || key.meta) return;
		if (key.return) {
			if (key.shift) setValue(current => `${current}\n`);
			else if (value.trim()) {
				void onSubmit(value);
				setValue('');
			}
			return;
		}
		if (key.backspace || key.delete) {
			setValue(current => current.slice(0, -1));
			return;
		}
		if (input) setValue(current => current + input);
	});
	return <Box borderStyle="round" paddingX={1}><Text color="blue">&gt; </Text><Text>{value || (busy ? 'busy - wait for the current run to finish' : '')}</Text></Box>;
}

function ApprovalModal({approval, onDecision, onCancel}: {approval: Approval; onDecision: (decision: 'allow' | 'deny') => Promise<void>; onCancel: () => Promise<void>}): React.ReactNode {
	useInput((input, key) => {
		if (input.toLowerCase() === 'y') void onDecision('allow');
		else if (input.toLowerCase() === 'n') void onDecision('deny');
		else if (key.escape) void onCancel();
	});
	return <Box borderStyle="double" borderColor="yellow" flexDirection="column" paddingX={1} marginTop={1}>
		<Text bold color="yellow">Approval required</Text>
		<Text>Tool: {approval.toolName}</Text>
		{approval.command ? <Text>Command: {approval.command}</Text> : null}
		{approval.path ? <Text>File: {approval.path}</Text> : null}
		{approval.cwd ? <Text>cwd: {approval.cwd}</Text> : null}
		<Text>Reason: {approval.policyReason}</Text>
		{approval.toolName === 'shell' ? <Text color="yellow">Shell commands are not sandboxed and may access external resources.</Text> : null}
		<Text>[Y] Allow  [N] Deny  [Esc] Cancel turn</Text>
	</Box>;
}

function SessionPicker({sessions, onResume, onClose}: {sessions: SessionSummary[]; onResume: (sessionId: string) => Promise<void>; onClose: () => void}): React.ReactNode {
	const [selected, setSelected] = useState(0);
	useInput((input, key) => {
		if (key.escape) onClose();
		else if (key.upArrow) setSelected(index => Math.max(0, index - 1));
		else if (key.downArrow) setSelected(index => Math.min(sessions.length - 1, index + 1));
		else if (key.return && sessions[selected]) void onResume(sessions[selected].session_id);
	});
	return <Box borderStyle="double" borderColor="cyan" flexDirection="column" paddingX={1} marginTop={1}>
		<Text bold>Sessions</Text>
		{sessions.length === 0 ? <Text dimColor>No persisted sessions.</Text> : sessions.map((session, index) => <Text key={session.session_id} color={index === selected ? 'cyan' : undefined}>
			{index === selected ? '&gt; ' : '  '}{new Date(session.updated_at).toLocaleString()}  {session.first_user_preview ?? '(no user message)'}  {session.session_id.slice(0, 8)}
		</Text>)}
		<Text dimColor>↑↓ select · Enter resume · Esc close</Text>
	</Box>;
}

render(<App/>, TUI_RENDER_OPTIONS);

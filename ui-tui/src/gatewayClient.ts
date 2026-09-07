import {spawn, type ChildProcessWithoutNullStreams} from 'node:child_process';
import {randomUUID} from 'node:crypto';

export type RuntimeStatus = {
	model: string;
	workspace: string | null;
	web_enabled: boolean;
	permission_mode: 'ask' | 'full';
	session_id: string | null;
	recovery: {
		recovered_count: number;
		side_effects_unknown_count: number;
	};
	skill_count: number;
	terminal_backend: {
		kind: string;
		executor: string;
		cwd: string;
		is_filesystem_sandboxed: boolean;
	} | null;
	mcp: {
		server_states: Record<string, string>;
		issue_count: number;
	} | null;
	sandbox: {
		environment_kind: 'sandbox';
		state: string;
		sandbox_id: string;
		changed_path_count: number | null;
		apply_recovery_required: boolean;
		host_isolation_active: boolean;
		container_recreated_on_resume: boolean;
	} | null;
};

export type SessionSummary = {
	session_id: string;
	created_at: string;
	updated_at: string;
	first_user_preview: string | null;
};

export type TranscriptItem = {
	role: 'user' | 'assistant' | 'tool';
	text?: string;
	tool_name?: string;
	status?: string;
	summary?: string;
};

export type GatewayEvent = {
	type: string;
	payload: Record<string, unknown>;
};

type RpcResponse = {
	id?: string;
	result?: unknown;
	error?: {code: number; message: string};
	method?: string;
	params?: {type: string; payload: Record<string, unknown>};
};

export class GatewayClient {
	private readonly child: ChildProcessWithoutNullStreams;
	private readonly pending = new Map<string, {resolve: (value: unknown) => void; reject: (error: Error) => void}>();
	private readonly listeners = new Set<(event: GatewayEvent) => void>();
	private stdoutBuffer = '';
	private closed = false;
	private readonly closeOnProcessExit = (): void => this.endGatewayInput();

	public constructor() {
		const python = process.env.ROVA_TUI_PYTHON;
		if (!python) {
			throw new Error('ROVA_TUI_PYTHON is required to start the Rova TUI gateway.');
		}
		this.child = spawn(python, ['-m', 'rova.app.tui_gateway'], {
			cwd: process.cwd(),
			env: process.env,
			shell: false,
			stdio: 'pipe',
		});
		this.child.stdout.setEncoding('utf8');
		this.child.stdout.on('data', (chunk: string) => this.consumeStdout(chunk));
		this.child.stderr.pipe(process.stderr);
		this.child.on('error', error => this.rejectPending(error));
		this.child.on('exit', (code, signal) => {
			this.rejectPending(new Error(`Rova gateway exited (${code ?? 'unknown'}${signal ? `, ${signal}` : ''}).`));
		});
		process.once('exit', this.closeOnProcessExit);
	}

	public onEvent(listener: (event: GatewayEvent) => void): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	public async request<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
		const id = randomUUID();
		const message = JSON.stringify({jsonrpc: '2.0', id, method, params});
		return new Promise<T>((resolve, reject) => {
			this.pending.set(id, {resolve: value => resolve(value as T), reject});
			this.child.stdin.write(`${message}\n`, error => {
				if (error) {
					this.pending.delete(id);
					reject(error);
				}
			});
		});
	}

	public async close(): Promise<void> {
		if (this.closed) return;
		this.closed = true;
		process.removeListener('exit', this.closeOnProcessExit);
		try {
			await this.request('runtime.close');
		} catch {
			// Gateway EOF is fail-closed for pending approvals; ending stdin still requests cleanup.
		} finally {
			this.endGatewayInput();
		}
	}

	private endGatewayInput(): void {
		if (!this.child.stdin.destroyed) this.child.stdin.end();
	}

	private consumeStdout(chunk: string): void {
		this.stdoutBuffer += chunk;
		let newlineIndex = this.stdoutBuffer.indexOf('\n');
		while (newlineIndex !== -1) {
			const line = this.stdoutBuffer.slice(0, newlineIndex);
			this.stdoutBuffer = this.stdoutBuffer.slice(newlineIndex + 1);
			if (line) this.consumeFrame(line);
			newlineIndex = this.stdoutBuffer.indexOf('\n');
		}
	}

	private consumeFrame(line: string): void {
		let frame: RpcResponse;
		try {
			frame = JSON.parse(line) as RpcResponse;
		} catch {
			this.rejectPending(new Error('Rova gateway emitted invalid JSON-RPC.'));
			return;
		}
		if (frame.method === 'event' && frame.params) {
			for (const listener of this.listeners) listener(frame.params);
			return;
		}
		if (!frame.id) return;
		const pending = this.pending.get(frame.id);
		if (!pending) return;
		this.pending.delete(frame.id);
		if (frame.error) pending.reject(new Error(frame.error.message));
		else pending.resolve(frame.result);
	}

	private rejectPending(error: Error): void {
		for (const pending of this.pending.values()) pending.reject(error);
		this.pending.clear();
	}
}

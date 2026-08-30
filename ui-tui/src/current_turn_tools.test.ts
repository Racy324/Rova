import assert from 'node:assert/strict';

import {applyCurrentTurnToolEvent, beginCurrentTurn} from './current_turn_tools.js';

const resumedToolActivity = [
	{id: 'old-call', toolName: 'read', summary: 'paper.pdf', status: 'success' as const},
];

const activeTurn = beginCurrentTurn(resumedToolActivity);
assert.deepEqual(activeTurn, []);

const started = applyCurrentTurnToolEvent(activeTurn, {
	type: 'tool.start',
	id: 'new-call',
	toolName: 'shell',
	summary: 'python -V',
});
assert.deepEqual(started, [
	{id: 'new-call', toolName: 'shell', summary: 'python -V', status: 'running'},
]);

assert.deepEqual(applyCurrentTurnToolEvent(started, {
	type: 'tool.end',
	id: 'new-call',
	status: 'success',
}), [
	{id: 'new-call', toolName: 'shell', summary: 'python -V', status: 'success'},
]);

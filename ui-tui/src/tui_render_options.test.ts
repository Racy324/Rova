import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

import {TUI_RENDER_OPTIONS} from './tui_render_options.js';

assert.equal(TUI_RENDER_OPTIONS.alternateScreen, true);
assert.equal(TUI_RENDER_OPTIONS.exitOnCtrlC, false);
assert.equal(TUI_RENDER_OPTIONS.interactive, true);

const packageJson = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')) as {
	dependencies: {ink: string};
};

assert.match(packageJson.dependencies.ink, /^\^7\./);

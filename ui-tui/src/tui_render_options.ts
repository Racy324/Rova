export const TUI_RENDER_OPTIONS = {
	exitOnCtrlC: false,
	// The Python launcher inherits the user's terminal, but Node can report a
	// falsy stdout.isTTY under Windows terminal hosts. This is an interactive
	// TUI entrypoint, so make Ink retain its cursor/erase lifecycle instead of
	// falling back to append-only output.
	interactive: true,
	alternateScreen: true,
} as const;

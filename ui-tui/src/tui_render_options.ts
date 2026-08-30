export const TUI_RENDER_OPTIONS = {
	exitOnCtrlC: false,
	// entry.tsx supplies Ink with a TTY-shaped forwarding stdout because Node can
	// report a falsy stdout.isTTY under Windows terminal hosts. Keep redraws in
	// the primary terminal buffer so users retain normal scrollback history.
	interactive: true,
	alternateScreen: false,
} as const;

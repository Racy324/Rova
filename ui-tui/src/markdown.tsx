import React from 'react';
import {Box, Text} from 'ink';

function InlineText({text}: {text: string}): React.ReactNode {
	const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
	return parts.map((part, index) => {
		if (part.startsWith('**') && part.endsWith('**')) return <Text key={index} bold>{part.slice(2, -2)}</Text>;
		if (part.startsWith('`') && part.endsWith('`')) return <Text key={index} color="cyan">{part.slice(1, -1)}</Text>;
		return <React.Fragment key={index}>{part}</React.Fragment>;
	});
}

/** A deliberately small Ink presentation layer, not a general Markdown parser. */
export function Markdown({text}: {text: string}): React.ReactNode {
	const lines = text.split('\n');
	const rendered: React.ReactNode[] = [];
	let codeLines: string[] = [];
	let inCode = false;
	for (const [index, line] of lines.entries()) {
		if (line.startsWith('```')) {
			if (inCode) {
				rendered.push(<Box key={`code-${index}`} borderStyle="round" paddingX={1}><Text color="cyan">{codeLines.join('\n')}</Text></Box>);
				codeLines = [];
			}
			inCode = !inCode;
			continue;
		}
		if (inCode) {
			codeLines.push(line);
			continue;
		}
		if (line.startsWith('# ')) rendered.push(<Text key={index} bold color="magenta">{line.slice(2)}</Text>);
		else if (line.startsWith('## ')) rendered.push(<Text key={index} bold>{line.slice(3)}</Text>);
		else if (/^[-*]\s+/.test(line)) rendered.push(<Text key={index}>• <InlineText text={line.replace(/^[-*]\s+/, '')}/></Text>);
		else rendered.push(<Text key={index}><InlineText text={line}/></Text>);
	}
	if (codeLines.length) rendered.push(<Box key="unterminated-code" borderStyle="round" paddingX={1}><Text color="cyan">{codeLines.join('\n')}</Text></Box>);
	return <Box flexDirection="column">{rendered}</Box>;
}

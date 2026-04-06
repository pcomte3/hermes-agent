"""Streaming and non-streaming Markdown rendering for the Hermes CLI.

Provides:
- ``MarkdownStreamProcessor``: stateful line-by-line markdown-to-ANSI
  transformer for the streaming display path.
- ``render_markdown_rich``: one-shot Rich Markdown renderable for the
  non-streaming Panel path.
"""

from __future__ import annotations

import re
import shutil
from typing import Optional

# ---------------------------------------------------------------------------
# ANSI escape helpers
# ---------------------------------------------------------------------------

_RST = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"
_UNDERLINE = "\033[4m"
_STRIKETHROUGH = "\033[9m"
_BOLD_ITALIC = "\033[1;3m"
_INLINE_CODE = "\033[1;36;40m"  # cyan bold on dark bg

# Sentinel Unicode chars (private-use area) for escaped markdown characters.
# These are substituted *before* regex transforms and restored *after*.
_ESC_STAR = "\ue000"
_ESC_UNDER = "\ue001"
_ESC_TILDE = "\ue002"
_ESC_BACKTICK = "\ue003"

_ESCAPE_MAP = {
    r"\*": _ESC_STAR,
    r"\_": _ESC_UNDER,
    r"\~": _ESC_TILDE,
    r"\`": _ESC_BACKTICK,
}
_RESTORE_MAP = {v: k[1] for k, v in _ESCAPE_MAP.items()}  # sentinel -> literal char


# ---------------------------------------------------------------------------
# Inline markdown regex transforms
# ---------------------------------------------------------------------------

def _build_inline_transforms():
    """Build compiled regex transforms for inline markdown.

    Returns a list of (compiled_regex, replacement_or_callable) tuples.
    Applied in order to each normal-mode line.
    """
    transforms = []

    # Headers: # through #### at start of line
    transforms.append((
        re.compile(r"^(#{1,4})\s+(.+)$"),
        "_header",
    ))

    # Horizontal rule: --- or *** or ___ (3+ of the same char, optionally spaced)
    transforms.append((
        re.compile(r"^[\s]*[-*_]{3,}[\s]*$"),
        "_hr",
    ))

    # Bold+italic: ***text*** or ___text___
    transforms.append((
        re.compile(r"(\*{3}|_{3})(?!\s)(.+?)(?<!\s)\1"),
        "_bold_italic",
    ))

    # Bold: **text** or __text__
    transforms.append((
        re.compile(r"(\*{2}|_{2})(?!\s)(.+?)(?<!\s)\1"),
        "_bold",
    ))

    # Italic: *text* (not inside words)
    transforms.append((
        re.compile(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)"),
        "_italic_star",
    ))

    # Italic: _text_ (not inside words)
    transforms.append((
        re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)"),
        "_italic_under",
    ))

    # Strikethrough: ~~text~~
    transforms.append((
        re.compile(r"~~(?!\s)(.+?)(?<!\s)~~"),
        "_strikethrough",
    ))

    # Inline code: `code` (but NOT ``` fences)
    transforms.append((
        re.compile(r"(?<!`)`(?!`)([^`]+)`(?!`)"),
        "_inline_code",
    ))

    # Links: [text](url)
    transforms.append((
        re.compile(r"\[([^\]]+)\]\(([^)]+)\)"),
        "_link",
    ))

    # Blockquote: > text at start of line
    transforms.append((
        re.compile(r"^(\s*>\s?)(.*)$"),
        "_blockquote",
    ))

    # Unordered list: -, *, + at start of line
    transforms.append((
        re.compile(r"^(\s*)([-*+])\s"),
        "_ul_marker",
    ))

    # Ordered list: 1. 2. etc. at start of line
    transforms.append((
        re.compile(r"^(\s*)(\d+\.)\s"),
        "_ol_marker",
    ))

    return transforms


_INLINE_TRANSFORMS = _build_inline_transforms()

# Pipe-table row: starts and ends with | (after stripping), has at least one
# inner |.  Also matches separator rows like |---|---|.
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|.*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:?-]+(\|[\s:?-]+)+\|\s*$")


# ---------------------------------------------------------------------------
# MarkdownStreamProcessor
# ---------------------------------------------------------------------------

class MarkdownStreamProcessor:
    """Stateful line-by-line markdown-to-ANSI transformer for streaming output.

    Designed to sit between ``_emit_stream_text``'s line buffer and ``_cprint``.
    Processes one complete line at a time.  The caller handles line buffering
    (which ``_emit_stream_text`` already does).

    Two states:
        NORMAL     – apply inline markdown regex transforms
        CODE_BLOCK – highlight each line with Pygments
    """

    def __init__(self, base_ansi: str = "", pygments_theme: str = "monokai"):
        """
        Args:
            base_ansi: Base text color ANSI escape (from skin's banner_text).
                       Applied after every reset to restore the "default" color.
            pygments_theme: Pygments style name for code blocks.
        """
        self._base = base_ansi
        self._rst = _RST
        self._theme = pygments_theme

        # State
        self._in_code_block = False
        self._code_lang = ""
        self._code_lines: list[str] = []
        self._lexer = None
        self._formatter = None
        self._md_fence_depth = 0  # depth of skipped ```markdown/```md fences
        self._table_rows: list[str] = []  # buffered pipe-table rows

    # -- public API ---------------------------------------------------------

    def feed_line(self, line: str) -> Optional[str]:
        """Process one complete line and return ANSI-formatted output.

        Returns ``None`` when the line is buffered (e.g. table rows) and
        should NOT be printed.  The caller must skip ``_cprint`` for ``None``.
        """
        stripped = line.lstrip()

        # -- Table buffering ---------------------------------------------------
        # Pipe-table rows are buffered until a non-table line arrives, then
        # the whole table is rendered at once (needs all rows for column widths).
        if not self._in_code_block and _TABLE_ROW_RE.match(line):
            self._table_rows.append(line)
            return None  # suppress — will be emitted when the table ends

        # If we were buffering a table and this line is NOT a table row,
        # flush the table first, then process this line normally.
        table_output = ""
        if self._table_rows:
            table_output = self._render_table()
            self._table_rows.clear()

        # -- Fence detection: ``` at start of line (after optional whitespace) --
        if stripped.startswith("```"):
            if not self._in_code_block:
                # Opening fence — extract language
                lang = stripped[3:].strip().split()[0] if stripped[3:].strip() else ""

                # Skip ```markdown / ```md fences — LLMs often wrap entire
                # responses in these.  Render the inner content as normal
                # markdown instead of treating it as a code block.
                if lang.lower() in ("markdown", "md"):
                    self._md_fence_depth += 1
                    return table_output or ""

                # Bare ``` in NORMAL mode: if we previously skipped a
                # ```markdown opener, this is its matching closer — skip it.
                if not lang and self._md_fence_depth > 0:
                    self._md_fence_depth -= 1
                    return table_output or ""

                self._enter_code_block(lang)
                fence = self._render_fence_header(lang)
                return f"{table_output}\n{fence}" if table_output else fence
            else:
                # Closing fence
                self._exit_code_block()
                return self._render_fence_footer()

        if self._in_code_block:
            self._code_lines.append(line)
            return self._highlight_code_line(line)

        result = self._apply_inline_markdown(line)
        return f"{table_output}\n{result}" if table_output else result

    def flush(self) -> Optional[str]:
        """Finalize state.  Returns any pending output (table, code block)."""
        parts = []
        if self._table_rows:
            parts.append(self._render_table())
            self._table_rows.clear()
        if self._in_code_block:
            self._exit_code_block()
            parts.append(self._render_fence_footer())
        return "\n".join(parts) if parts else None

    def reset(self):
        """Reset all state for a new response."""
        self._in_code_block = False
        self._code_lang = ""
        self._code_lines.clear()
        self._lexer = None
        self._formatter = None
        self._md_fence_depth = 0
        self._table_rows.clear()

    # -- code block internals -----------------------------------------------

    def _enter_code_block(self, lang: str):
        self._in_code_block = True
        self._code_lang = lang or "text"
        self._code_lines.clear()
        self._lexer = None
        self._formatter = None

    def _exit_code_block(self):
        self._in_code_block = False
        self._code_lang = ""
        self._code_lines.clear()
        self._lexer = None
        self._formatter = None

    def _get_lexer(self):
        if self._lexer is None:
            from pygments.lexers import get_lexer_by_name, TextLexer
            try:
                self._lexer = get_lexer_by_name(self._code_lang, stripall=False)
            except Exception:
                self._lexer = TextLexer()
        return self._lexer

    def _get_formatter(self):
        if self._formatter is None:
            from pygments.formatters import TerminalTrueColorFormatter
            self._formatter = TerminalTrueColorFormatter(style=self._theme)
        return self._formatter

    def _highlight_code_line(self, line: str) -> str:
        from pygments import highlight as _pyg_highlight
        lexer = self._get_lexer()
        formatter = self._get_formatter()
        # Highlight the single line; Pygments adds a trailing newline — strip it
        result = _pyg_highlight(line + "\n", lexer, formatter).rstrip("\n")
        return f"  {result}{self._rst}"

    def _render_fence_header(self, lang: str) -> str:
        w = shutil.get_terminal_size((80, 24)).columns
        label = f" {lang} " if lang else ""
        fill = max(w - 6 - len(label), 3)
        return f"{_DIM}  ┌───{self._rst}{_DIM}{_BOLD}{label}{self._rst}{_DIM}{'─' * fill}{self._rst}"

    def _render_fence_footer(self) -> str:
        w = shutil.get_terminal_size((80, 24)).columns
        fill = max(w - 4, 3)
        return f"{_DIM}  └{'─' * fill}{self._rst}"

    # -- table internals ----------------------------------------------------

    def _render_table(self) -> str:
        """Render buffered pipe-table rows as a box-drawing table."""
        base = self._base
        rst = self._rst

        # Parse rows into cells
        parsed: list[list[str]] = []
        sep_indices: set[int] = set()
        for i, raw in enumerate(self._table_rows):
            row = raw.strip()
            # Strip leading/trailing pipes and split
            if row.startswith("|"):
                row = row[1:]
            if row.endswith("|"):
                row = row[:-1]
            cells = [c.strip() for c in row.split("|")]
            parsed.append(cells)
            if _TABLE_SEP_RE.match(self._table_rows[i]):
                sep_indices.add(i)

        if not parsed:
            return ""

        # Compute column widths using terminal-aware width (handles emoji,
        # CJK characters, etc. that occupy 2 cells in the terminal).
        try:
            from wcwidth import wcswidth
            def _display_width(s: str) -> int:
                w = wcswidth(s)
                return w if w >= 0 else len(s)
        except ImportError:
            _display_width = len

        n_cols = max(len(r) for r in parsed)
        col_widths = [0] * n_cols
        for i, row in enumerate(parsed):
            if i in sep_indices:
                continue  # don't let separator dashes inflate widths
            for j, cell in enumerate(row):
                if j < n_cols:
                    col_widths[j] = max(col_widths[j], _display_width(cell))

        # Ensure minimum width of 3 per column
        col_widths = [max(w, 3) for w in col_widths]

        def _pad_cell(val: str, target_width: int) -> str:
            """Pad a cell value to target_width using display-aware spacing."""
            current = _display_width(val)
            pad = max(target_width - current, 0)
            return val + " " * pad

        def _border(left: str, mid: str, right: str, fill: str = "─") -> str:
            segs = [fill * (w + 2) for w in col_widths]
            return f"{_DIM}{left}{mid.join(segs)}{right}{rst}"

        lines = []
        lines.append(_border("  ┌", "┬", "┐"))

        for i, row in enumerate(parsed):
            if i in sep_indices:
                # Render separator as a mid-border
                lines.append(_border("  ├", "┼", "┤"))
            else:
                # Bold the header row (row 0, if followed by a separator)
                is_header = (i == 0 and 1 in sep_indices)
                cell_strs = []
                for j in range(n_cols):
                    val = row[j] if j < len(row) else ""
                    padded = _pad_cell(val, col_widths[j])
                    cell_strs.append(
                        f" {_BOLD if is_header else ''}{base}{padded}{rst} "
                    )
                inner = f"{_DIM}│{rst}".join(cell_strs)
                lines.append(f"  {_DIM}│{rst}{inner}{_DIM}│{rst}")

        lines.append(_border("  └", "┴", "┘"))
        return "\n".join(lines)

    # -- inline markdown internals ------------------------------------------

    def _apply_inline_markdown(self, line: str) -> str:
        base = self._base
        rst = self._rst

        # Escape backslash-escaped markdown characters
        for esc_seq, sentinel in _ESCAPE_MAP.items():
            line = line.replace(esc_seq, sentinel)

        for pattern, handler_name in _INLINE_TRANSFORMS:
            if handler_name == "_header":
                m = pattern.match(line)
                if m:
                    level = len(m.group(1))
                    text = m.group(2)
                    # h1 = bold+underline, h2 = bold, h3/h4 = bold+dim
                    if level == 1:
                        line = f"{_BOLD}{_UNDERLINE}{text}{rst}{base}"
                    elif level == 2:
                        line = f"{_BOLD}{text}{rst}{base}"
                    else:
                        line = f"{_BOLD}{_DIM}{text}{rst}{base}"
                    continue

            elif handler_name == "_hr":
                if pattern.match(line):
                    w = shutil.get_terminal_size((80, 24)).columns
                    line = f"{_DIM}{'─' * min(w - 4, 40)}{rst}{base}"
                    continue

            elif handler_name == "_bold_italic":
                line = pattern.sub(
                    lambda m: f"{_BOLD_ITALIC}{m.group(2)}{rst}{base}", line
                )

            elif handler_name == "_bold":
                line = pattern.sub(
                    lambda m: f"{_BOLD}{m.group(2)}{rst}{base}", line
                )

            elif handler_name == "_italic_star":
                line = pattern.sub(
                    lambda m: f"{_ITALIC}{m.group(1)}{rst}{base}", line
                )

            elif handler_name == "_italic_under":
                line = pattern.sub(
                    lambda m: f"{_ITALIC}{m.group(1)}{rst}{base}", line
                )

            elif handler_name == "_strikethrough":
                line = pattern.sub(
                    lambda m: f"{_STRIKETHROUGH}{m.group(1)}{rst}{base}", line
                )

            elif handler_name == "_inline_code":
                line = pattern.sub(
                    lambda m: f"{_INLINE_CODE}{m.group(1)}{rst}{base}", line
                )

            elif handler_name == "_link":
                line = pattern.sub(
                    lambda m: f"{_UNDERLINE}{m.group(1)}{rst}{_DIM} ({m.group(2)}){rst}{base}",
                    line,
                )

            elif handler_name == "_blockquote":
                m = pattern.match(line)
                if m:
                    text = m.group(2)
                    line = f"{_DIM}  │ {_ITALIC}{text}{rst}{base}"

            elif handler_name == "_ul_marker":
                m = pattern.match(line)
                if m:
                    indent = m.group(1)
                    rest = line[m.end():]
                    line = f"{indent}  • {rest}"

            elif handler_name == "_ol_marker":
                m = pattern.match(line)
                if m:
                    indent = m.group(1)
                    num = m.group(2)
                    rest = line[m.end():]
                    line = f"{indent}  {_DIM}{num}{rst}{base} {rest}"

        # Restore escaped characters
        for sentinel, literal in _RESTORE_MAP.items():
            line = line.replace(sentinel, literal)

        # Apply base color wrapping
        if base:
            return f"{base}{line}{rst}"
        return line


# ---------------------------------------------------------------------------
# Non-streaming helper
# ---------------------------------------------------------------------------

def render_markdown_rich(text: str, pygments_theme: str = "monokai"):
    """Return a Rich Markdown renderable for the non-streaming Panel path.

    Args:
        text: Complete LLM response text (plain markdown).
        pygments_theme: Pygments style name for fenced code blocks.

    Returns:
        A ``rich.markdown.Markdown`` instance suitable for ``Panel(content)``.
    """
    from rich.markdown import Markdown
    return Markdown(text, code_theme=pygments_theme)

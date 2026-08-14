//! A line editor with history, for the interactive shell.
//!
//! In canonical mode the kernel edits the line for us, and does a poor job:
//! backspace works, nothing else does. An arrow key is not "move the cursor",
//! it is the three bytes `ESC [ D` deposited into the buffer, which is why
//! pressing Up at an unprepared prompt prints `^[[A`. Getting real editing
//! means taking the terminal out of that mode (see `platform::tty::RawMode`)
//! and doing the work here.
//!
//! # Why this module owns no terminal
//!
//! Everything below reads bytes from an [`io::Read`] and writes bytes to an
//! [`io::Write`]. It never enables raw mode, never touches file descriptor 0,
//! and contains no `unsafe` — the caller wraps the call in a `RawMode` guard
//! and passes stdin/stdout in.
//!
//! That is what makes a line editor testable at all. Every key sequence, edit
//! operation, and history interaction below is exercised against a `&[u8]` and
//! a `Vec<u8>` in the test module, with no TTY and no subprocess. A design that
//! read directly from fd 0 could only be tested by hand.
//!
//! # Deliberate limits
//!
//! - **Width is counted in characters, not display columns.** Same trade-off
//!   the table renderer makes: full width handling needs a Unicode table, and
//!   that is a dependency (CLAUDE.md §2).
//! - **A line longer than the terminal is redrawn, not reflowed.** Cursor
//!   placement past the wrap point drifts. Fixing it properly needs the
//!   terminal width and wrap-aware redraw; statements that long are better
//!   typed across several lines, which the shell already supports.
//! - **A lone `ESC` waits for the next keystroke** rather than timing out,
//!   because distinguishing it from the start of `ESC [ A` requires a timed
//!   read. Pressing Escape then Enter behaves like Enter.

use std::collections::VecDeque;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;

/// How many entries the history file keeps. Old ones are dropped from the
/// front on save.
const MAX_HISTORY: usize = 1000;

/// The result of asking the user for one line.
#[derive(Debug, PartialEq, Eq)]
pub enum Input {
    /// A completed line, without its trailing newline.
    Line(String),
    /// Ctrl-C. The line being typed is abandoned; the session is not.
    Interrupted,
    /// Ctrl-D on an empty line, or the input stream ending.
    Eof,
}

// ---------------------------------------------------------------------------
// Keys — bytes in, keystrokes out
// ---------------------------------------------------------------------------

/// One decoded keystroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Key {
    Char(char),
    Enter,
    Backspace,
    Delete,
    Left,
    Right,
    WordLeft,
    WordRight,
    Home,
    End,
    Up,
    Down,
    /// Ctrl-C.
    Interrupt,
    /// Ctrl-D.
    Eof,
    /// Ctrl-K.
    KillToEnd,
    /// Ctrl-U.
    KillToStart,
    /// Ctrl-W.
    DeleteWord,
    /// Ctrl-L.
    ClearScreen,
    /// Recognized as a key, deliberately ignored (an unmapped escape
    /// sequence, a stray control byte, invalid UTF-8).
    Ignored,
}

/// A byte source that hands out keystrokes.
///
/// Owns its buffer for the life of the session rather than per line: in raw
/// mode a paste arrives as one burst of bytes spanning several lines, and a
/// reader that dropped its buffer at the end of a line would lose everything
/// typed ahead.
pub struct Keys<R: Read> {
    input: R,
    buf: [u8; 256],
    /// Bytes read but not yet consumed, as `[pos, len)`.
    pos: usize,
    len: usize,
    /// Bytes pushed back during escape-sequence decoding.
    pending: VecDeque<u8>,
}

impl<R: Read> Keys<R> {
    /// Wrap a byte source.
    pub fn new(input: R) -> Self {
        Keys {
            input,
            buf: [0; 256],
            pos: 0,
            len: 0,
            pending: VecDeque::new(),
        }
    }

    /// The next byte, or `None` at end of input.
    fn byte(&mut self) -> io::Result<Option<u8>> {
        if let Some(b) = self.pending.pop_front() {
            return Ok(Some(b));
        }
        while self.pos == self.len {
            match self.input.read(&mut self.buf) {
                Ok(0) => return Ok(None),
                Ok(n) => {
                    self.pos = 0;
                    self.len = n;
                }
                // EINTR. The SIGINT handler is installed without SA_RESTART
                // precisely so reads return here — but in raw mode ISIG is off
                // and Ctrl-C arrives as a byte, so any signal that lands here
                // is something else and the read simply resumes.
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(Some(b))
    }

    /// Un-read a byte, so a partially-decoded escape sequence can give back
    /// what turned out not to belong to it.
    fn unread(&mut self, b: u8) {
        self.pending.push_back(b);
    }

    /// Decode the next keystroke, or `None` at end of input.
    fn key(&mut self) -> io::Result<Option<Key>> {
        let Some(b) = self.byte()? else {
            return Ok(None);
        };
        let key = match b {
            0x01 => Key::Home,        // Ctrl-A
            0x02 => Key::Left,        // Ctrl-B
            0x03 => Key::Interrupt,   // Ctrl-C
            0x04 => Key::Eof,         // Ctrl-D
            0x05 => Key::End,         // Ctrl-E
            0x06 => Key::Right,       // Ctrl-F
            0x08 => Key::Backspace,   // Ctrl-H
            0x09 => Key::Ignored,     // Tab: no completion to offer yet
            0x0a | 0x0d => Key::Enter, // NL / CR — ICRNL is off, so Enter is CR
            0x0b => Key::KillToEnd,   // Ctrl-K
            0x0c => Key::ClearScreen, // Ctrl-L
            0x0e => Key::Down,        // Ctrl-N
            0x10 => Key::Up,          // Ctrl-P
            0x15 => Key::KillToStart, // Ctrl-U
            0x17 => Key::DeleteWord,  // Ctrl-W
            0x1b => self.escape()?,
            0x7f => Key::Backspace, // what most terminals actually send
            // Remaining C0 controls are unmapped; passing them through would
            // put unprintable bytes in a statement.
            b if b < 0x20 => Key::Ignored,
            // Printable ASCII — the overwhelmingly common case, and a single
            // byte by definition.
            b if b < 0x80 => Key::Char(b as char),
            b => match self.utf8(b)? {
                Some(c) => Key::Char(c),
                None => Key::Ignored,
            },
        };
        Ok(Some(key))
    }

    /// Finish a multi-byte UTF-8 character whose leading byte is `first`.
    ///
    /// Returns `None` for invalid UTF-8 rather than erroring: a mistyped byte
    /// on a terminal is not a failure of the shell, and swallowing it is
    /// better than ending the session over it.
    fn utf8(&mut self, first: u8) -> io::Result<Option<char>> {
        let extra = match first {
            0xc2..=0xdf => 1,
            0xe0..=0xef => 2,
            0xf0..=0xf4 => 3,
            _ => return Ok(None),
        };
        let mut bytes = vec![first];
        for _ in 0..extra {
            match self.byte()? {
                Some(b) if (0x80..0xc0).contains(&b) => bytes.push(b),
                // Not a continuation byte: it belongs to the NEXT keystroke,
                // so give it back rather than eating it.
                Some(b) => {
                    self.unread(b);
                    return Ok(None);
                }
                None => return Ok(None),
            }
        }
        Ok(std::str::from_utf8(&bytes)
            .ok()
            .and_then(|s| s.chars().next()))
    }

    /// Decode what follows an `ESC`.
    ///
    /// Handles the CSI (`ESC [`) and SS3 (`ESC O`) forms terminals use for
    /// arrows, Home/End and Delete. Anything else — Alt-chords, unknown
    /// sequences — is consumed whole and ignored, which is what keeps a stray
    /// sequence from spraying its bytes into the statement being typed.
    fn escape(&mut self) -> io::Result<Key> {
        let Some(intro) = self.byte()? else {
            return Ok(Key::Ignored);
        };
        if intro != b'[' && intro != b'O' {
            return Ok(Key::Ignored); // Alt-<key> and friends
        }

        // A CSI sequence is parameter bytes then one final byte in 0x40..=0x7e.
        let mut params = Vec::new();
        let final_byte = loop {
            match self.byte()? {
                Some(b) if (0x40..=0x7e).contains(&b) => break b,
                Some(b) => params.push(b),
                None => return Ok(Key::Ignored),
            }
            if params.len() > 16 {
                return Ok(Key::Ignored); // malformed; stop consuming
            }
        };

        // Ctrl-modified arrows arrive as `ESC [ 1 ; 5 C`. Word-wise movement
        // is the one modified form worth mapping — it is how people jump over
        // a column name they mistyped.
        let ctrl = params.ends_with(b";5");
        Ok(match (final_byte, ctrl) {
            (b'C', true) => Key::WordRight,
            (b'D', true) => Key::WordLeft,
            (b'A', _) => Key::Up,
            (b'B', _) => Key::Down,
            (b'C', _) => Key::Right,
            (b'D', _) => Key::Left,
            (b'H', _) => Key::Home,
            (b'F', _) => Key::End,
            // Matched WHOLE, not by first byte: `ESC [ 1 ~` is Home but
            // `ESC [ 1 5 ~` is F5, and a prefix match would turn every
            // function key into a cursor jump.
            (b'~', _) => match params.as_slice() {
                b"1" | b"7" => Key::Home,
                b"3" => Key::Delete,
                b"4" | b"8" => Key::End,
                _ => Key::Ignored,
            },
            _ => Key::Ignored,
        })
    }
}

// ---------------------------------------------------------------------------
// Line — the buffer being edited
// ---------------------------------------------------------------------------

/// The text of the line and where the cursor sits in it.
///
/// `cursor` is a BYTE index that is always on a `char` boundary. Bytes rather
/// than characters because every operation here slices the string, and
/// characters rather than a raw offset because a UTF-8 statement must not be
/// splittable mid-character.
#[derive(Default)]
struct Line {
    buf: String,
    cursor: usize,
}

impl Line {
    /// Replace the whole line and park the cursor at the end — what history
    /// recall does.
    fn set(&mut self, text: &str) {
        self.buf.clear();
        self.buf.push_str(text);
        self.cursor = self.buf.len();
    }

    fn insert(&mut self, c: char) {
        self.buf.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    /// Byte index of the char boundary before the cursor.
    fn prev_boundary(&self) -> Option<usize> {
        self.buf[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(i, _)| i)
    }

    /// Byte index of the char boundary after the cursor.
    fn next_boundary(&self) -> Option<usize> {
        self.buf[self.cursor..]
            .chars()
            .next()
            .map(|c| self.cursor + c.len_utf8())
    }

    fn backspace(&mut self) {
        if let Some(i) = self.prev_boundary() {
            self.buf.remove(i);
            self.cursor = i;
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.buf.len() {
            self.buf.remove(self.cursor);
        }
    }

    fn left(&mut self) {
        if let Some(i) = self.prev_boundary() {
            self.cursor = i;
        }
    }

    fn right(&mut self) {
        if let Some(i) = self.next_boundary() {
            self.cursor = i;
        }
    }

    fn home(&mut self) {
        self.cursor = 0;
    }

    fn end(&mut self) {
        self.cursor = self.buf.len();
    }

    fn kill_to_end(&mut self) {
        self.buf.truncate(self.cursor);
    }

    fn kill_to_start(&mut self) {
        self.buf.replace_range(..self.cursor, "");
        self.cursor = 0;
    }

    /// Start of the word at or before the cursor: skip whitespace, then skip
    /// the word itself.
    fn word_start(&self) -> usize {
        let mut i = self.cursor;
        let before = |i: usize| self.buf[..i].chars().next_back();
        while let Some(c) = before(i) {
            if c.is_whitespace() {
                i -= c.len_utf8();
            } else {
                break;
            }
        }
        while let Some(c) = before(i) {
            if c.is_whitespace() {
                break;
            }
            i -= c.len_utf8();
        }
        i
    }

    /// End of the word at or after the cursor.
    fn word_end(&self) -> usize {
        let mut i = self.cursor;
        let after = |i: usize| self.buf[i..].chars().next();
        while let Some(c) = after(i) {
            if c.is_whitespace() {
                i += c.len_utf8();
            } else {
                break;
            }
        }
        while let Some(c) = after(i) {
            if c.is_whitespace() {
                break;
            }
            i += c.len_utf8();
        }
        i
    }

    fn delete_word(&mut self) {
        let start = self.word_start();
        self.buf.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Cursor position in CHARACTERS, which is what a terminal column is.
    fn column(&self) -> usize {
        self.buf[..self.cursor].chars().count()
    }
}

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

/// Reads lines with editing and history.
pub struct Editor {
    history: Vec<String>,
    /// Where history is persisted, if a home directory was found.
    path: Option<PathBuf>,
}

impl Editor {
    /// Build an editor, loading history from `$HOME/.flats_history`.
    ///
    /// A missing or unreadable history file is not an error — it is what the
    /// first run looks like.
    pub fn new() -> Self {
        let path = std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".flats_history"));
        let history = path
            .as_ref()
            .and_then(|p| fs::read_to_string(p).ok())
            .map(|text| text.lines().map(str::to_string).collect())
            .unwrap_or_default();
        Editor { history, path }
    }

    /// Write history back to disk, keeping the most recent [`MAX_HISTORY`].
    ///
    /// Best effort: a shell that refused to exit because it could not write a
    /// convenience file would be worse than one that quietly loses history.
    pub fn save(&self) {
        let Some(path) = &self.path else { return };
        let start = self.history.len().saturating_sub(MAX_HISTORY);
        let mut text = self.history[start..].join("\n");
        text.push('\n');
        let _ = fs::write(path, text);
    }

    /// Record a line in history.
    ///
    /// Blank lines and immediate repeats are dropped — pressing Enter twice,
    /// or re-running the same query, should not push the interesting entries
    /// further away from the Up arrow.
    fn remember(&mut self, line: &str) {
        if line.trim().is_empty() {
            return;
        }
        if self.history.last().map(String::as_str) == Some(line) {
            return;
        }
        self.history.push(line.to_string());
    }

    /// Read one line, echoing and editing it via `out`.
    ///
    /// The caller is responsible for having put the terminal in raw mode; see
    /// the module header for why that split exists.
    pub fn read_line<R: Read, W: Write>(
        &mut self,
        prompt: &str,
        keys: &mut Keys<R>,
        out: &mut W,
    ) -> io::Result<Input> {
        let mut line = Line::default();
        // Where Up/Down currently sit in history, and the line that was being
        // typed before navigation started, so Down can bring it back.
        let mut browsing: Option<usize> = None;
        let mut stashed = String::new();

        redraw(out, prompt, &line)?;

        loop {
            let Some(key) = keys.key()? else {
                // The stream ended mid-line. Treat a half-typed line as
                // abandoned rather than submitting something never confirmed.
                return Ok(Input::Eof);
            };

            match key {
                Key::Enter => {
                    // The newline is written here, not left to the caller, so
                    // the cursor leaves the prompt line before anything else
                    // prints — otherwise output lands on top of the prompt.
                    out.write_all(b"\n")?;
                    out.flush()?;
                    let text = std::mem::take(&mut line.buf);
                    self.remember(&text);
                    return Ok(Input::Line(text));
                }
                Key::Interrupt => {
                    // `^C` is echoed because raw mode does not: without it,
                    // Ctrl-C would silently blank the line and look like a
                    // dropped keystroke.
                    out.write_all(b"^C\n")?;
                    out.flush()?;
                    return Ok(Input::Interrupted);
                }
                Key::Eof => {
                    if line.buf.is_empty() {
                        out.write_all(b"\n")?;
                        out.flush()?;
                        return Ok(Input::Eof);
                    }
                    // Ctrl-D mid-line is the readline convention for
                    // delete-forward, and only means EOF on an empty line.
                    line.delete();
                }
                Key::Char(c) => line.insert(c),
                Key::Backspace => line.backspace(),
                Key::Delete => line.delete(),
                Key::Left => line.left(),
                Key::Right => line.right(),
                Key::WordLeft => line.cursor = line.word_start(),
                Key::WordRight => line.cursor = line.word_end(),
                Key::Home => line.home(),
                Key::End => line.end(),
                Key::KillToEnd => line.kill_to_end(),
                Key::KillToStart => line.kill_to_start(),
                Key::DeleteWord => line.delete_word(),
                Key::ClearScreen => {
                    // Home the cursor, erase the screen, then let the redraw
                    // below put the prompt back at the top.
                    out.write_all(b"\x1b[H\x1b[2J")?;
                }
                Key::Up => {
                    let next = match browsing {
                        // First Up: stash what is being typed, then land on
                        // the most recent entry.
                        None if !self.history.is_empty() => {
                            stashed = line.buf.clone();
                            Some(self.history.len() - 1)
                        }
                        Some(0) | None => browsing, // already at the oldest
                        Some(i) => Some(i - 1),
                    };
                    if next != browsing {
                        browsing = next;
                        if let Some(i) = browsing {
                            line.set(&self.history[i]);
                        }
                    }
                }
                Key::Down => match browsing {
                    Some(i) if i + 1 < self.history.len() => {
                        browsing = Some(i + 1);
                        line.set(&self.history[i + 1]);
                    }
                    // Past the newest entry: back to the line that was
                    // interrupted by the first Up.
                    Some(_) => {
                        browsing = None;
                        let stash = std::mem::take(&mut stashed);
                        line.set(&stash);
                    }
                    None => {}
                },
                Key::Ignored => {}
            }

            redraw(out, prompt, &line)?;
        }
    }
}

/// Repaint the prompt and line, then place the cursor.
///
/// The whole line is rewritten on every keystroke rather than patched. At
/// human typing speed the cost is irrelevant, and a diffing redraw is where
/// line editors accumulate their display bugs.
fn redraw(out: &mut impl Write, prompt: &str, line: &Line) -> io::Result<()> {
    // `\r` to column zero, then the content, then erase whatever the previous
    // (longer) line left behind — without the erase, deleting a character
    // leaves its ghost on screen.
    write!(out, "\r{prompt}{}\x1b[K\r", line.buf)?;
    let column = prompt.chars().count() + line.column();
    if column > 0 {
        // CUF with an explicit count. Guarded because `ESC [ 0 C` still moves
        // one column on some terminals.
        write!(out, "\x1b[{column}C")?;
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::{Editor, Input, Key, Keys, Line};

    /// Drive `read_line` over a fixed byte script, returning what it decided
    /// and what it painted.
    fn read(editor: &mut Editor, input: &[u8]) -> (Input, String) {
        let mut keys = Keys::new(input);
        let mut out = Vec::new();
        let result = editor
            .read_line("> ", &mut keys, &mut out)
            .expect("a slice never fails to read");
        (result, String::from_utf8(out).expect("output is utf-8"))
    }

    /// An editor with no history file, so tests never touch `$HOME`.
    fn editor() -> Editor {
        Editor {
            history: Vec::new(),
            path: None,
        }
    }

    fn line_of(text: &str, cursor: usize) -> Line {
        Line {
            buf: text.to_string(),
            cursor,
        }
    }

    // -- key decoding --------------------------------------------------------

    fn keys_of(bytes: &[u8]) -> Vec<Key> {
        let mut keys = Keys::new(bytes);
        let mut out = Vec::new();
        while let Some(k) = keys.key().expect("a slice never fails to read") {
            out.push(k);
        }
        out
    }

    #[test]
    fn arrow_keys_decode_as_movement_not_text() {
        // The bug this module exists to fix: in canonical mode these three
        // bytes end up IN the statement as `^[[A`.
        assert_eq!(
            keys_of(b"\x1b[A\x1b[B\x1b[C\x1b[D"),
            [Key::Up, Key::Down, Key::Right, Key::Left]
        );
    }

    #[test]
    fn home_end_and_delete_decode_in_both_forms() {
        // Terminals disagree about these: xterm sends `ESC [ H`, others send
        // `ESC [ 1 ~`. Both have to work or the keys break on half of them.
        assert_eq!(keys_of(b"\x1b[H\x1b[F"), [Key::Home, Key::End]);
        assert_eq!(keys_of(b"\x1b[1~\x1b[4~"), [Key::Home, Key::End]);
        assert_eq!(keys_of(b"\x1b[3~"), [Key::Delete]);
        // SS3 form, sent when the terminal is in application cursor mode.
        assert_eq!(keys_of(b"\x1bOA"), [Key::Up]);
    }

    #[test]
    fn ctrl_arrows_decode_as_word_movement() {
        assert_eq!(keys_of(b"\x1b[1;5C\x1b[1;5D"), [Key::WordRight, Key::WordLeft]);
    }

    #[test]
    fn control_bytes_map_to_editing_commands() {
        assert_eq!(
            keys_of(b"\x01\x05\x03\x04\x0b\x15\x17\x0c"),
            [
                Key::Home,
                Key::End,
                Key::Interrupt,
                Key::Eof,
                Key::KillToEnd,
                Key::KillToStart,
                Key::DeleteWord,
                Key::ClearScreen,
            ]
        );
        // Both backspace encodings — terminals send 0x7f, Ctrl-H sends 0x08.
        assert_eq!(keys_of(b"\x7f\x08"), [Key::Backspace, Key::Backspace]);
        // CR, because ICRNL is off in raw mode, and NL for piped input.
        assert_eq!(keys_of(b"\r\n"), [Key::Enter, Key::Enter]);
    }

    #[test]
    fn an_unmapped_escape_sequence_is_swallowed_whole() {
        // The point is that its BYTES do not reach the line: an unrecognized
        // F5 must not type `[15~` into a query.
        assert_eq!(keys_of(b"\x1b[15~"), [Key::Ignored]);
        assert_eq!(keys_of(b"\x1b[15~a"), [Key::Ignored, Key::Char('a')]);
    }

    #[test]
    fn multi_byte_characters_decode_as_one_key() {
        assert_eq!(keys_of("é".as_bytes()), [Key::Char('é')]);
        assert_eq!(keys_of("日本".as_bytes()), [Key::Char('日'), Key::Char('本')]);
        assert_eq!(keys_of("aé漢".as_bytes()), [
            Key::Char('a'),
            Key::Char('é'),
            Key::Char('漢')
        ]);
    }

    #[test]
    fn invalid_utf8_is_ignored_without_eating_the_next_key() {
        // A truncated sequence followed by a real key: the key must survive,
        // which is what `unread` is for.
        assert_eq!(keys_of(b"\xc3a"), [Key::Ignored, Key::Char('a')]);
        assert_eq!(keys_of(b"\xff"), [Key::Ignored]);
    }

    // -- editing operations --------------------------------------------------

    #[test]
    fn insertion_happens_at_the_cursor() {
        let mut line = line_of("ac", 1);
        line.insert('b');
        assert_eq!(line.buf, "abc");
        assert_eq!(line.cursor, 2, "cursor follows the inserted character");
    }

    #[test]
    fn movement_and_deletion_respect_char_boundaries() {
        // `é` is two bytes. Stepping by one byte would split it and panic on
        // the next slice.
        let mut line = line_of("aéb", 4);
        line.left();
        assert_eq!(line.cursor, 3);
        line.left();
        assert_eq!(line.cursor, 1, "one keypress crosses the whole character");
        line.right();
        assert_eq!(line.cursor, 3);

        let mut line = line_of("aéb", 3);
        line.backspace();
        assert_eq!(line.buf, "ab");
        assert_eq!(line.cursor, 1);
    }

    #[test]
    fn kill_commands_cut_on_the_right_side_of_the_cursor() {
        let mut line = line_of("SELECT * FROM t", 9);
        line.kill_to_end();
        assert_eq!(line.buf, "SELECT * ");

        let mut line = line_of("SELECT * FROM t", 9);
        line.kill_to_start();
        assert_eq!(line.buf, "FROM t");
        assert_eq!(line.cursor, 0);
    }

    #[test]
    fn delete_word_removes_the_word_before_the_cursor() {
        let mut line = line_of("SELECT * FROM docs", 18);
        line.delete_word();
        assert_eq!(line.buf, "SELECT * FROM ");
        // Trailing whitespace is skipped before the word is found, so a
        // cursor sitting after a space still eats the word, not just the gap.
        let mut line = line_of("SELECT * FROM docs  ", 20);
        line.delete_word();
        assert_eq!(line.buf, "SELECT * FROM ");
    }

    #[test]
    fn word_movement_stops_at_word_edges() {
        let line = line_of("SELECT * FROM docs", 18);
        assert_eq!(line.word_start(), 14);
        let line = line_of("SELECT * FROM docs", 0);
        assert_eq!(line.word_end(), 6);
    }

    #[test]
    fn the_column_is_counted_in_characters_not_bytes() {
        // Cursor placement uses this; counting bytes would put the cursor two
        // columns off for every accented character typed.
        assert_eq!(line_of("aé", 3).column(), 2);
    }

    // -- reading a line ------------------------------------------------------

    #[test]
    fn a_typed_line_comes_back_without_its_terminator() {
        let (input, _) = read(&mut editor(), b"SELECT 1;\r");
        assert_eq!(input, Input::Line("SELECT 1;".to_string()));
    }

    #[test]
    fn editing_keys_change_the_line_that_is_returned() {
        // Mistype `SELEK`, backspace over the `K`, finish the word — the
        // returned text is the EDITED text, which is the whole point.
        let (input, _) = read(&mut editor(), b"SELEK\x7fCT\r");
        assert_eq!(input, Input::Line("SELECT".to_string()));

        // Left-arrow then insert.
        let (input, _) = read(&mut editor(), b"ac\x1b[Db\r");
        assert_eq!(input, Input::Line("abc".to_string()));

        // Home, then insert at the front.
        let (input, _) = read(&mut editor(), b"b\x01a\r");
        assert_eq!(input, Input::Line("ab".to_string()));
    }

    #[test]
    fn ctrl_c_abandons_the_line_and_ctrl_d_ends_the_session() {
        let (input, painted) = read(&mut editor(), b"SELECT\x03");
        assert_eq!(input, Input::Interrupted, "the line is abandoned");
        assert!(painted.contains("^C"), "and the cancel is visible");

        // Ctrl-D on an EMPTY line is EOF ...
        assert_eq!(read(&mut editor(), b"\x04").0, Input::Eof);
        // ... but mid-line it deletes forward instead.
        let (input, _) = read(&mut editor(), b"ab\x01\x04\r");
        assert_eq!(input, Input::Line("b".to_string()));
    }

    #[test]
    fn an_unterminated_line_at_end_of_input_is_not_submitted() {
        // Nothing confirmed it. Submitting half a statement because the pipe
        // closed would run something the user never pressed Enter on.
        assert_eq!(read(&mut editor(), b"SELECT 1").0, Input::Eof);
    }

    // -- history -------------------------------------------------------------

    #[test]
    fn up_recalls_the_previous_line_and_down_returns() {
        let mut editor = editor();
        read(&mut editor, b"first;\r");
        read(&mut editor, b"second;\r");

        // One Up lands on the most recent entry.
        assert_eq!(
            read(&mut editor, b"\x1b[A\r").0,
            Input::Line("second;".to_string())
        );
        // Two Ups reach the older one.
        assert_eq!(
            read(&mut editor, b"\x1b[A\x1b[A\r").0,
            Input::Line("first;".to_string())
        );
        // Up then Down comes back to where it started.
        assert_eq!(
            read(&mut editor, b"\x1b[A\x1b[B\r").0,
            Input::Line(String::new())
        );
    }

    #[test]
    fn a_recalled_line_can_be_edited_before_running() {
        let mut editor = editor();
        read(&mut editor, b"SELECT 1;\r");
        // Recall, then backspace over `;` and append.
        let (input, _) = read(&mut editor, b"\x1b[A\x7f, 2;\r");
        assert_eq!(input, Input::Line("SELECT 1, 2;".to_string()));
    }

    #[test]
    fn history_navigation_preserves_the_line_being_typed() {
        let mut editor = editor();
        read(&mut editor, b"old;\r");
        // Type something, look at history, come back: the draft is still there.
        let (input, _) = read(&mut editor, b"draft\x1b[A\x1b[B\r");
        assert_eq!(input, Input::Line("draft".to_string()));
    }

    #[test]
    fn blank_lines_and_repeats_are_not_recorded() {
        let mut editor = editor();
        read(&mut editor, b"\r");
        read(&mut editor, b"   \r");
        assert!(editor.history.is_empty(), "blank lines stay out of history");

        read(&mut editor, b"SELECT 1;\r");
        read(&mut editor, b"SELECT 1;\r");
        assert_eq!(editor.history, ["SELECT 1;"], "a repeat is not re-recorded");
    }

    #[test]
    fn down_at_the_bottom_of_history_does_nothing() {
        // Regression guard for the index arithmetic: pressing Down without
        // having pressed Up must not underflow or resurrect an entry.
        let mut editor = editor();
        read(&mut editor, b"first;\r");
        let (input, _) = read(&mut editor, b"\x1b[B\x1b[Bnew\r");
        assert_eq!(input, Input::Line("new".to_string()));
    }

    #[test]
    fn up_at_the_top_of_history_stays_put() {
        let mut editor = editor();
        read(&mut editor, b"only;\r");
        // Five Ups with one entry: still that entry, no panic.
        let (input, _) = read(&mut editor, b"\x1b[A\x1b[A\x1b[A\x1b[A\x1b[A\r");
        assert_eq!(input, Input::Line("only;".to_string()));
    }

    #[test]
    fn up_with_empty_history_leaves_the_line_alone() {
        let (input, _) = read(&mut editor(), b"ab\x1b[A\r");
        assert_eq!(input, Input::Line("ab".to_string()));
    }
}

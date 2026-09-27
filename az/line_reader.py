"""Byte-level JSON-lines reader for engine pipes, shared by the agent runners.

Engine messages can exceed the pipe buffer and several can arrive in one chunk, so neither a
plain buffered readline() (blocks with no cap) nor select() + buffered readline() works: the
buffered reader may swallow the NEXT line into its internal buffer while select() no longer sees
data on the file descriptor — if that line is a query the engine now waits for our reply and
both sides block (see engine_bridge._read for the same pattern).
"""

from __future__ import annotations

import os
import select
import time

READ_TIMEOUT = 60.0


class LineReader:
    """Byte-level line assembler for the engine stdout with a hard cap per line.

    Battle states can exceed the pipe buffer, so a plain buffered readline() is not enough:
    a raw read may swallow a partial line into the internal buffer while select() no longer
    sees any data on the file descriptor (see engine_bridge._read for the same pattern).
    """

    def __init__( self, fd: int ):
        self._fd = fd
        self._buffer = b""

    def read_line( self, timeout: float = READ_TIMEOUT ) -> str | None:
        """Returns one line without the newline, or None on EOF."""
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError( "engine did not produce output within the time limit" )

            ready, _, _ = select.select( [self._fd], [], [], min( remaining, 1.0 ) )
            if not ready:
                continue

            chunk = os.read( self._fd, 65536 )
            if not chunk:
                line, self._buffer = self._buffer, b""
                return line.decode() if line.strip() else None
            self._buffer += chunk

        line, self._buffer = self._buffer.split( b"\n", 1 )
        return line.decode()

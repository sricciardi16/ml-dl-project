import sys
import io
import os
import tempfile
import subprocess
import atexit


class ReportGenerator:
    """
    Generates markdown reports by capturing standard output and embedding diagrams.

    This class must be instantiated with a target markdown file path (`md_file`).
    It redirects `sys.stdout` to capture `print()` statements and writes them into
    the markdown file as code blocks.

    Public API:
        - `__init__(self, md_file)`: Initializes the generator and starts capturing stdout.
        - `kroki(self, obj, title=None)`: Converts and embeds a matplotlib Figure or graphviz Dot object.
        - `close(self)`: Flushes buffers, restores stdout, and cleans up.

    .. warning::
        Methods like `write`, `flush`, and any starting with an underscore (e.g., `_append_to_md`,
        `_flush_buffer`, `_send_to_kroki`) are strictly for internal stream handling and
        must not be used publicly.
    """

    def __init__(self, md_file):
        """
        Initializes the ReportGenerator with a target markdown file path.
        """
        self.md_file = md_file
        self.terminal = sys.stdout
        # String buffer to temporarily hold captured output
        self.buffer = io.StringIO()
        # Markdown block flags to track if we are currently inside a ```text block
        self._in_text_block = False
        # Flag to track if a newline is needed before closing the markdown block
        self._needs_newline = False
        # Hijack sys.stdout to redirect all print statements to this instance
        sys.stdout = self
        # Register the close method to run at program exit to ensure cleanup
        atexit.register(self.close)

    def __enter__(self):
        # Enable context manager usage by returning self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Enable context manager usage by ensuring close is called on exit
        self.close()

    def __getattr__(self, name):
        return getattr(self.terminal, name)

    def write(self, message):
        """
        Internal override for sys.stdout.write. Do not call this method directly.
        """
        self.terminal.write(message)
        self.terminal.flush()
        self.buffer.write(message)
        if message:
            with open(self.md_file, "a", encoding="utf-8") as f:
                # If we are not currently in a text block, open a new markdown code block
                if not self._in_text_block:
                    f.write("```text\n")
                    self._in_text_block = True
                f.write(message)
                self._needs_newline = not message.endswith("\n")

    def flush(self):
        """
        Internal override for sys.stdout.flush. Do not call this method directly.
        """
        self.terminal.flush()

    def _append_to_md(self, text):
        with open(self.md_file, "a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n\n")

    def _flush_buffer(self):
        # If we are currently in a text block, close the markdown block
        if self._in_text_block:
            with open(self.md_file, "a", encoding="utf-8") as f:
                if self._needs_newline:
                    f.write("\n")
                f.write("```\n\n")
            self._in_text_block = False
            self._needs_newline = False
        # Clear the io.StringIO buffer and reset its position
        self.buffer.seek(0)
        self.buffer.truncate(0)

    def _send_to_kroki(self, code, diagram_type):
        # Create a temporary file to write the diagram code
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(code)
            temp_file_name = temp_file.name

        try:
            # Call the Kroki CLI subprocess to encode the diagram
            result = subprocess.run(
                ["kroki", "-encode", temp_file_name, "-type", diagram_type],
                capture_output=True,
                text=True,
                check=True
            )
            encoded = result.stdout.rstrip()
            self._append_to_md(encoded)
        except subprocess.CalledProcessError as e:
            print(f"Kroki error: {e.stderr}", file=self.terminal)
        finally:
            # Ensure the temporary file is cleaned up in the finally block
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)

    def kroki(self, obj, title=None):
        """
        Converts a matplotlib Figure or graphviz Dot object and embeds the SVG/diagram.

        This method accepts a matplotlib Figure or graphviz Dot object, converts it
        to the appropriate format (SVG for matplotlib, source text for graphviz),
        and sends it to Kroki to generate and embed the diagram in the markdown file.

        Args:
            obj: A matplotlib Figure or graphviz Dot object.
            title (str, optional): An optional title to write before the diagram.

        Raises:
            ValueError: If the object type is not supported.
        """
        self._flush_buffer()

        if title is not None:
            self._append_to_md(f"**{title}**")

        class_name = type(obj).__name__
        if class_name == "Figure":
            buffer = io.StringIO()
            obj.savefig(buffer, format='svg')
            code = buffer.getvalue()
            buffer.close()
            diagram_type = "svgdraw"
        elif class_name == "Dot":
            code = obj.source if hasattr(obj, 'source') else obj.to_string()
            diagram_type = "graphviz"
        else:
            raise ValueError(f"Unsupported object type: {class_name}")

        self._send_to_kroki(code, diagram_type)

    def close(self):
        # Flush any remaining text in the buffer and close the markdown block
        self._flush_buffer()
        # Restore sys.stdout to the original terminal stream
        if sys.stdout is self:
            sys.stdout = self.terminal
            # Unregister the atexit handler to prevent redundant calls
            atexit.unregister(self.close)
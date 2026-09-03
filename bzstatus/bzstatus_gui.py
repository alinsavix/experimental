"""
Backblaze Status GUI
A simple GUI application to monitor Backblaze backup status
"""

import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import scrolledtext, ttk

from backblaze_client import BackblazeClient


class BackblazeStatusGUI:
    def __init__(self, root, debug=False):
        self.root = root
        self.debug = debug
        self.root.title("Backblaze Backup Status Monitor")
        self.root.geometry("900x700")

        self.bz_client = BackblazeClient()
        self.refresh_interval = 10000  # 10 seconds
        self.size_fetch_active = False  # Track if size fetching is active
        self.file_size_cache = {}  # Cache file sizes by path
        self.last_file_mtime = None  # Track last modification time of bzlist file
        self._sort_column = None
        self._sort_reverse = False
        self._all_files = []  # Sample of the pending list; the tree shows a filtered view
        self._last_status_info = {}
        self._testimony = {}  # Backblaze's own backlog totals (see parse_host_testimony)

        self.setup_ui()
        self.update_status()

    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Status Section
        status_frame = ttk.LabelFrame(main_frame, text="Current Status", padding="10")
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)

        # Status labels
        ttk.Label(status_frame, text="Client Status:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.status_label = ttk.Label(status_frame, text="Unknown", font=('Arial', 10, 'bold'))
        self.status_label.grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(status_frame, text="Activity:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.activity_label = ttk.Label(status_frame, text="None", foreground="blue")
        self.activity_label.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Label(status_frame, text="Current File:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.current_file_label = ttk.Label(status_frame, text="None", wraplength=600, justify=tk.LEFT)
        self.current_file_label.grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Label(status_frame, text="Progress:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.progress_label = ttk.Label(status_frame, text="0%")
        self.progress_label.grid(row=3, column=1, sticky=tk.W, pady=2)

        ttk.Label(status_frame, text="Total Pending:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.pending_label = ttk.Label(status_frame, text="0 files")
        self.pending_label.grid(row=4, column=1, sticky=tk.W, pady=2)

        ttk.Label(status_frame, text="Last Updated:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.last_update_label = ttk.Label(status_frame, text="Never")
        self.last_update_label.grid(row=5, column=1, sticky=tk.W, pady=2)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.refresh_button = ttk.Button(button_frame, text="Refresh Now", command=self.manual_refresh)
        self.refresh_button.pack(side=tk.LEFT, padx=5)

        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Auto-refresh",
                       variable=self.auto_refresh_var).pack(side=tk.LEFT, padx=5)

        # Refresh interval selector
        ttk.Label(button_frame, text="Interval:").pack(side=tk.LEFT, padx=(10, 2))
        self.interval_var = tk.StringVar(value="10s")
        interval_combo = ttk.Combobox(button_frame, textvariable=self.interval_var,
                                     values=["5s", "10s", "30s", "1m", "5m", "10m"],
                                     width=8, state="readonly")
        interval_combo.pack(side=tk.LEFT, padx=2)
        interval_combo.bind("<<ComboboxSelected>>", self.on_interval_changed)

        # File-watch mode
        self.file_watch_mode = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Only on file change",
                       variable=self.file_watch_mode).pack(side=tk.LEFT, padx=5)

        # Files list section
        files_frame = ttk.LabelFrame(main_frame, text="Files Scheduled for Backup", padding="10")
        files_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(1, weight=1)

        # Search / filter box
        search_frame = ttk.Frame(files_frame)
        search_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        search_frame.columnconfigure(1, weight=1)

        ttk.Label(search_frame, text="Search:").grid(row=0, column=0, padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace_add('write', lambda *_: self._populate_tree())
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.search_entry.bind('<Escape>', lambda e: self.search_var.set(''))
        ttk.Button(search_frame, text="Clear", width=6,
                   command=lambda: self.search_var.set('')).grid(row=0, column=2, padx=(5, 0))
        self.filter_label = ttk.Label(search_frame, text="", foreground="gray")
        self.filter_label.grid(row=0, column=3, padx=(8, 0))

        # Ctrl+F from anywhere jumps to the search box
        self.root.bind('<Control-f>', lambda e: (self.search_entry.focus_set(),
                                                 self.search_entry.select_range(0, tk.END)))

        # Create Treeview for files
        columns = ('filename', 'size', 'status')
        self.tree = ttk.Treeview(files_frame, columns=columns, show='headings', height=15)

        self.tree.heading('filename', text='File Path', command=lambda: self._sort_tree('filename'))
        self.tree.heading('size', text='Size', command=lambda: self._sort_tree('size'))
        self.tree.heading('status', text='Status', command=lambda: self._sort_tree('status'))

        self.tree.column('filename', width=500)
        self.tree.column('size', width=100)
        self.tree.column('status', width=150)

        # Scrollbars
        vsb = ttk.Scrollbar(files_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(files_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=1, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=2, column=0, sticky=(tk.W, tk.E))

        # Copy support
        self.tree.bind('<Control-c>', lambda e: self._copy_selected_paths())
        self._context_menu = tk.Menu(self.tree, tearoff=0)
        self._context_menu.add_command(label="Copy path(s)", command=self._copy_selected_paths)
        self.tree.bind('<Button-3>', self._show_context_menu)

        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))

    def format_size(self, size_bytes):
        """Format file size in human-readable format"""
        if size_bytes is None:
            return "Unknown"

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                fmt = ".0f" if unit in ('B', 'KB') else ".2f"
                return f"{size_bytes:{fmt}} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def on_interval_changed(self, event=None):
        """Handle refresh interval change"""
        interval_str = self.interval_var.get()
        # Parse interval string
        if interval_str.endswith('s'):
            seconds = int(interval_str[:-1])
        elif interval_str.endswith('m'):
            seconds = int(interval_str[:-1]) * 60
        else:
            seconds = 10  # Default

        self.refresh_interval = seconds * 1000  # Convert to milliseconds

    def should_refresh(self):
        """Check if we should refresh based on file-watch mode"""
        if not self.file_watch_mode.get():
            return True  # Normal auto-refresh mode

        # File-watch mode: only refresh if bzlist_filesremaining.txt changed
        import os
        bzlist_path = os.path.join(
            self.bz_client.bz_data_path or 'C:\\ProgramData\\Backblaze',
            'bzdata', 'bzreports', 'bzlist_filesremaining.txt'
        )

        try:
            if os.path.exists(bzlist_path):
                current_mtime = os.path.getmtime(bzlist_path)
                if self.last_file_mtime is None or current_mtime != self.last_file_mtime:
                    self.last_file_mtime = current_mtime
                    return True
        except:
            pass

        return False

    def update_status(self):
        """Update the status display"""
        if not self.auto_refresh_var.get():
            # Schedule next update even if not refreshing
            self.root.after(self.refresh_interval, self.update_status)
            return

        # In file-watch mode, status always refreshes; only the file list is gated
        full_refresh = self.should_refresh()

        self.status_bar.config(text="Updating...")
        self.refresh_button.config(state='disabled')

        # Run update in separate thread to avoid blocking UI
        thread = threading.Thread(target=self.fetch_and_update, args=(full_refresh,), daemon=True)
        thread.start()

        # Set a watchdog timer - if update doesn't complete in 30 seconds, re-enable button
        self.root.after(30000, self.check_update_timeout)

    def check_update_timeout(self):
        """Check if an update is taking too long"""
        if self.refresh_button['state'] == 'disabled':
            self.refresh_button.config(state='normal')
            self.status_bar.config(text="Update timed out - click Refresh to try again")

    def fetch_and_update(self, full_refresh=True):
        """Fetch data from Backblaze and update UI"""
        try:
            # Always fetch status and totals; only fetch file list on full refresh
            status_info = self.bz_client.get_status()
            testimony = self.bz_client.parse_host_testimony()
            pending_files = self.bz_client.get_pending_files() if full_refresh else None

            # Schedule UI update on main thread
            self.root.after(0, lambda: self.update_ui(status_info, pending_files, testimony))

        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))

    def update_ui(self, status_info, pending_files, testimony=None):
        """Update UI elements with fetched data"""
        try:
            if testimony:
                self._testimony = testimony
            if self.debug:
                print(f"[DEBUG UI] Received status_info: {status_info}")

            # Update status labels
            client_status = status_info.get('status', 'Unknown')
            self.status_label.config(text=client_status.replace('_', ' '))

            # Color code the status (normalize underscores so e.g. 'not_running' == 'not running')
            status_norm = client_status.lower().replace('_', ' ')
            if status_norm in ['running', 'backing up', 'transmitting', 'uploading part']:
                self.status_label.config(foreground='green')
            elif status_norm.startswith('paused') or status_norm in ['preparing', 'preparing large file']:
                self.status_label.config(foreground='orange')
            elif status_norm in ['not running'] or status_norm.startswith('error'):
                self.status_label.config(foreground='red')
            else:
                # idle, sleeping, throttled, unknown, etc. — service is up, just not actively uploading
                self.status_label.config(foreground='gray')

            # Update activity label (shows the current file name from overviewstatus)
            current_file_name = status_info.get('current_file_name', '')
            if self.debug:
                print(f"[DEBUG UI] current_file_name: '{current_file_name}'")
            if status_info.get('paused'):
                self.activity_label.config(text=self._format_pause_activity(status_info))
            elif current_file_name:
                self.activity_label.config(text=current_file_name)
            else:
                self.activity_label.config(text='None')

            # Update current file (shows full path)
            current_file = status_info.get('current_file', 'None')
            if self.debug:
                print(f"[DEBUG UI] current_file: '{current_file}'")
            if current_file and current_file != 'none':
                self.current_file_label.config(text=current_file)
            else:
                self.current_file_label.config(text='None')

            # Update progress
            progress = status_info.get('progress', 0)
            self.progress_label.config(text=f"{progress}%")

            # Update last updated time
            self.last_update_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # Skip file list update if this was a status-only refresh (totals still refresh)
            if pending_files is None:
                self._update_pending_label()
                return

            self._all_files = pending_files
            self._last_status_info = status_info
            self._populate_tree()

            if pending_files:
                self.status_bar.config(text=f"Updated successfully at {datetime.now().strftime('%H:%M:%S')} - {len(pending_files)} file(s) found")
                # Start fetching sizes asynchronously
                self.fetch_file_sizes_async(pending_files)
            else:
                self.status_bar.config(text=f"Updated at {datetime.now().strftime('%H:%M:%S')} - No pending files detected")

        except Exception as e:
            self.show_error(f"Error updating UI: {str(e)}")
        finally:
            self.refresh_button.config(state='normal')
            # Schedule next update
            self.root.after(self.refresh_interval, self.update_status)

    def _format_pause_activity(self, status_info):
        """Describe an active pause: when it lifts and how long that is from now"""
        until_millis = status_info.get('pause_until_millis') or 0
        if not until_millis:
            return 'Paused'

        remaining_s = until_millis / 1000.0 - time.time()
        resumes_at = datetime.fromtimestamp(until_millis / 1000.0).strftime('%H:%M:%S')

        if remaining_s <= 0:
            return f'Pause expired at {resumes_at}'

        minutes = int(remaining_s // 60)
        if minutes >= 60:
            left = f'{minutes // 60}h {minutes % 60}m'
        else:
            left = f'{minutes}m'

        text = f'Resumes at {resumes_at} ({left} left)'
        # 'action_pause_backup' is the plain user-pressed-pause case; anything else is
        # worth surfacing verbatim since we haven't seen the full vocabulary
        reason = status_info.get('pause_reason', '')
        if reason and reason != 'action_pause_backup':
            text += f' — {reason}'
        return text

    def _populate_tree(self):
        """Rebuild the tree from self._all_files, honouring the current search filter"""
        # Every whitespace-separated term must appear in the path (case-insensitive)
        terms = self.search_var.get().lower().split()

        status_info = self._last_status_info
        current_uploading_file = status_info.get('current_file')
        current_file_name = status_info.get('current_file_name', '')

        for item in self.tree.get_children():
            self.tree.delete(item)

        shown = 0
        for file_info in self._all_files:
            filename = file_info.get('path', 'Unknown')
            if terms:
                lowered = filename.lower()
                if not all(term in lowered for term in terms):
                    continue

            # Use cached size if available, otherwise use what we have
            cached_size = self.file_size_cache.get(filename)
            current_size = file_info.get('size')

            # Prefer cached size if current is None; cache the file_info size if present
            if current_size is not None:
                self.file_size_cache[filename] = current_size
            display_size = current_size if current_size is not None else cached_size
            size_str = self.format_size(display_size)

            file_status = file_info.get('status', 'Pending')

            # Update status if this file is currently being uploaded
            if current_uploading_file and filename == current_uploading_file:
                # Determine more specific status based on current activity
                if current_file_name.startswith('Part '):
                    file_status = f'⬆ {current_file_name}'
                elif current_file_name.startswith('Preparing '):
                    file_status = '📝 Preparing for upload'
                else:
                    file_status = '⬆ Uploading now'

            self.tree.insert('', tk.END, values=(filename, size_str, file_status))
            shown += 1

        total = len(self._all_files)
        if not total:
            self.tree.insert('', tk.END, values=('No pending backup files found or unable to access Backblaze data', '-', 'N/A'))
            self.filter_label.config(text="")
        elif not shown:
            self.tree.insert('', tk.END, values=(f'No files match "{self.search_var.get()}"', '-', 'N/A'))
            self.filter_label.config(text=f"0 of {total}")
        elif terms:
            self.filter_label.config(text=f"{shown} of {total}")
        else:
            self.filter_label.config(text="")

        self._update_pending_label()

        # Re-apply sort if one is active
        self._apply_sort()

    def manual_refresh(self):
        """Manually trigger a refresh"""
        self.update_status()

    def fetch_file_sizes_async(self, pending_files):
        """Fetch file sizes asynchronously in background thread"""
        if self.size_fetch_active:
            return  # Already fetching sizes

        self.size_fetch_active = True
        thread = threading.Thread(target=self._fetch_sizes_worker, args=(pending_files,), daemon=True)
        thread.start()

    def _fetch_sizes_worker(self, pending_files):
        """Worker thread to fetch file sizes"""
        try:
            for i, file_info in enumerate(pending_files):
                file_path = file_info.get('path')
                if file_path:
                    # Check if we need to fetch (no size or not in cache)
                    if file_info.get('size') is None or file_path not in self.file_size_cache:
                        try:
                            import os
                            if os.path.exists(file_path):
                                size = os.path.getsize(file_path)
                                # Update cache
                                self.file_size_cache[file_path] = size
                                # Update UI on main thread (pass path, not index, so sorting doesn't misplace it)
                                self.root.after(0, self._update_file_size_in_tree, file_path, size)
                        except:
                            # If we can't get size, just skip it
                            pass
        finally:
            self.size_fetch_active = False

    def _update_file_size_in_tree(self, file_path, size_bytes):
        """Update the size column for the tree item matching file_path"""
        try:
            for item_id in self.tree.get_children():
                if self.tree.set(item_id, 'filename') == file_path:
                    values = self.tree.item(item_id, 'values')
                    if len(values) >= 3:
                        self.tree.item(item_id, values=(values[0], self.format_size(size_bytes), values[2]))
                        self._update_pending_label()
                    break
        except:
            pass  # Silently ignore errors if tree has been cleared

    def _update_pending_label(self):
        """Update the Total Pending label with the real backlog size

        Prefers Backblaze's own totals from bzhost_testimony_x.xml — these cover the whole
        backlog, whereas bzlist_filesremaining.txt is only a truncated sample of it. Falls
        back to summing what we can see if the testimony file is unavailable.
        """
        remaining = self._testimony.get('remaining_files')
        sample = len(self._all_files)

        if remaining is not None:
            total_bytes = self._testimony.get('remaining_bytes') or 0
            text = f"{remaining:,} files ({self.format_size(total_bytes)})"
            if sample and sample < remaining:
                text += f" — listing {sample}"
            self.pending_label.config(text=text)
            return

        # No testimony: fall back to the visible sample, flagged as a lower bound
        total = 0
        has_unknown = False
        for file_info in self._all_files:
            size = self.file_size_cache.get(file_info.get('path'))
            if size is not None:
                total += size
            else:
                has_unknown = True

        truncated = getattr(self.bz_client, 'list_truncated', False)
        count_str = f"{sample}+" if truncated else str(sample)
        prefix = '~' if (has_unknown or truncated) else ''
        self.pending_label.config(text=f"{count_str} files ({prefix}{self.format_size(total)})")

    def _sort_tree(self, col):
        """Sort the file list by the given column, toggling direction on repeated clicks"""
        if self._sort_column == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = col
            self._sort_reverse = False

        col_labels = {'filename': 'File Path', 'size': 'Size', 'status': 'Status'}
        for c, label in col_labels.items():
            if c == col:
                arrow = ' ▼' if self._sort_reverse else ' ▲'
                self.tree.heading(c, text=label + arrow)
            else:
                self.tree.heading(c, text=label)

        self._apply_sort()

    def _apply_sort(self):
        """Re-apply the current sort order to the tree"""
        if self._sort_column is None:
            return

        col = self._sort_column
        items = list(self.tree.get_children(''))

        if col == 'size':
            def size_key(k):
                filename = self.tree.set(k, 'filename')
                return self.file_size_cache.get(filename, -1)
            items.sort(key=size_key, reverse=self._sort_reverse)
        else:
            items.sort(key=lambda k: self.tree.set(k, col).lower(), reverse=self._sort_reverse)

        for index, k in enumerate(items):
            self.tree.move(k, '', index)

    def _copy_selected_paths(self):
        """Copy the file paths of selected rows to the clipboard"""
        selected = self.tree.selection()
        if not selected:
            return
        paths = [self.tree.set(item, 'filename') for item in selected]
        text = '\n'.join(paths)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_bar.config(text=f"Copied {len(paths)} path(s) to clipboard")

    def _show_context_menu(self, event):
        """Show right-click context menu on the tree"""
        # Select the row under the cursor if not already selected
        item = self.tree.identify_row(event.y)
        if item and item not in self.tree.selection():
            self.tree.selection_set(item)
        if self.tree.selection():
            self._context_menu.tk_popup(event.x_root, event.y_root)

    def show_error(self, error_msg):
        """Show error message"""
        self.status_bar.config(text=f"Error: {error_msg}")


def main():
    import sys
    debug = '--debug' in sys.argv

    # Set debug mode for BackblazeClient
    BackblazeClient.debug_mode = debug

    root = tk.Tk()
    app = BackblazeStatusGUI(root, debug=debug)
    root.mainloop()


if __name__ == "__main__":
    main()

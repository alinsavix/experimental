"""
Backblaze Client Interface
Handles communication with Backblaze backup client
"""

import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


class BackblazeClient:
    debug_mode = False  # Class variable for debug mode

    def __init__(self):
        """Initialize Backblaze client interface"""
        self.bz_executable = self.find_bz_executable()
        self.bz_data_path = self.find_bz_data_path()

    def find_bz_executable(self):
        """Find Backblaze CLI executable"""
        # Common Windows installation paths
        common_paths = [
            r"C:\Program Files (x86)\Backblaze\bzbui.exe",
            r"C:\Program Files\Backblaze\bzbui.exe",
            r"C:\Program Files (x86)\Backblaze\bzserv.exe",
            r"C:\Program Files\Backblaze\bzserv.exe",
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def find_bz_data_path(self):
        """Find Backblaze data directory"""
        # Common data paths
        common_paths = [
            os.path.join(os.environ.get('PROGRAMDATA', 'C:\\ProgramData'), 'Backblaze'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Backblaze'),
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        return None

    def run_command(self, command):
        """Run a Backblaze command and return output"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                shell=True
            )
            return result.stdout, result.stderr
        except Exception as e:
            return None, str(e)

    def parse_overviewstatus(self):
        """Parse overviewstatus.xml to get current transmission status"""
        status_info = {}

        try:
            if not self.bz_data_path:
                if self.debug_mode:
                    print("[DEBUG] No bz_data_path found")
                return status_info

            # Path to overviewstatus.xml
            overview_path = os.path.join(self.bz_data_path, 'bzdata', 'overviewstatus.xml')
            if self.debug_mode:
                print(f"[DEBUG] Looking for overviewstatus.xml at: {overview_path}")

            if not os.path.exists(overview_path):
                if self.debug_mode:
                    print(f"[DEBUG] overviewstatus.xml not found at {overview_path}")
                return status_info

            if self.debug_mode:
                print(f"[DEBUG] Reading overviewstatus.xml")
            # Parse the XML file
            tree = ET.parse(overview_path)
            root = tree.getroot()

            # Look for bztransmit element
            bztransmit = root.find('bztransmit')
            if bztransmit is not None:
                # Get current state
                cur_state = bztransmit.get('cur_state', 'unknown')
                current_file = bztransmit.get('current_file', '')
                current_file_fullpath = bztransmit.get('current_file_fullpath', '')
                millis_written = bztransmit.get('millisFileWasWritten', '')

                if self.debug_mode:
                    print(f"[DEBUG] cur_state: {cur_state}")
                    print(f"[DEBUG] current_file: {current_file}")
                    print(f"[DEBUG] current_file_fullpath: {current_file_fullpath}")

                # Determine status and current file info
                status_info['cur_state'] = cur_state
                status_info['current_file_name'] = current_file
                status_info['last_update_millis'] = millis_written

                # Always set the current file to the full path if available (not 'none')
                if current_file_fullpath and current_file_fullpath.lower() != 'none':
                    status_info['current_file'] = current_file_fullpath
                else:
                    status_info['current_file'] = None

                # Interpret the status
                if cur_state == 'transmitting':
                    if current_file == 'Producing File Lists...':
                        status_info['status'] = 'Preparing'
                    elif current_file.startswith('Preparing '):
                        status_info['status'] = 'Preparing Large File'
                    elif current_file.startswith('Part '):
                        status_info['status'] = 'Uploading Part'
                    elif current_file_fullpath and current_file_fullpath.lower() != 'none':
                        status_info['status'] = 'Backing Up'
                    else:
                        status_info['status'] = 'Transmitting'
                elif cur_state == 'not_running':
                    # bztransmit not active — service is running but idle between backup cycles
                    status_info['status'] = 'Idle'
                else:
                    status_info['status'] = cur_state.replace('_', ' ').capitalize()
            else:
                if self.debug_mode:
                    print("[DEBUG] bztransmit element not found in XML")

        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Error parsing overviewstatus.xml: {e}")
                import traceback
                traceback.print_exc()

        if self.debug_mode:
            print(f"[DEBUG] Returning status_info: {status_info}")
        return status_info

    def get_status(self):
        """Get current backup status"""
        status_info = {
            'status': 'Unknown',
            'current_file': None,
            'progress': 0
        }

        try:
            # Priority 1: Check overviewstatus.xml (most current)
            overview_status = self.parse_overviewstatus()
            if overview_status:
                status_info.update(overview_status)
                # If we found status from overviewstatus.xml, return it
                if 'status' in overview_status:
                    return status_info

            # Try to read status from bzdata.xml or bzreports.xml
            if self.bz_data_path:
                status_file = os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bzdatacenter.txt')
                reports_file = os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bzreports.xml')

                # Check various status files
                status_files = [
                    os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bzinfo.xml'),
                    os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bz_done.xml'),
                    reports_file
                ]

                for file_path in status_files:
                    if os.path.exists(file_path):
                        status_info.update(self.parse_status_file(file_path))
                        break

                # Try to get current operation from log files
                log_file = os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bz_done.txt')
                if os.path.exists(log_file):
                    current_file = self.parse_log_for_current_file(log_file)
                    if current_file:
                        status_info['current_file'] = current_file

            # If no data found, try to detect if service is running
            if status_info['status'] == 'Unknown':
                if self.is_service_running():
                    status_info['status'] = 'Running'
                else:
                    status_info['status'] = 'Not Running'

        except Exception as e:
            status_info['status'] = f'Error: {str(e)}'

        return status_info

    def is_service_running(self):
        """Check if Backblaze service is running"""
        try:
            result = subprocess.run(
                ['sc', 'query', 'Backblaze Backup Service'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'RUNNING' in result.stdout
        except:
            return False

    def parse_status_file(self, file_path):
        """Parse XML status file"""
        status = {}
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Try to extract status information
            for elem in root.iter():
                if 'state' in elem.tag.lower():
                    status['status'] = elem.text or 'Unknown'
                elif 'progress' in elem.tag.lower():
                    try:
                        status['progress'] = float(elem.text or 0)
                    except:
                        pass
                elif 'current' in elem.tag.lower() and 'file' in elem.tag.lower():
                    status['current_file'] = elem.text
        except:
            pass

        return status

    def parse_log_for_current_file(self, log_file):
        """Parse log file to find current file being backed up"""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                # Read last few lines
                lines = f.readlines()[-50:]
                for line in reversed(lines):
                    # Look for patterns indicating file being backed up
                    if 'backing up' in line.lower() or 'uploading' in line.lower():
                        # Extract file path
                        match = re.search(r'[A-Z]:\\[^:\n]+', line)
                        if match:
                            return match.group(0)
        except:
            pass
        return None

    def get_pending_files(self):
        """Get list of files pending backup"""
        pending_files = []

        try:
            if self.bz_data_path:
                # Priority 1: Backblaze's own files remaining report (most accurate!)
                files_remaining = os.path.join(self.bz_data_path, 'bzdata', 'bzreports', 'bzlist_filesremaining.txt')
                if os.path.exists(files_remaining):
                    if self.debug_mode:
                        print(f"[DEBUG] Reading pending files from: {files_remaining}")
                    pending_files = self.parse_pending_files(files_remaining)
                    if pending_files:
                        if self.debug_mode:
                            print(f"[DEBUG] Found {len(pending_files)} files from bzlist_filesremaining.txt")
                        return pending_files

                # Priority 2: Look for explicit todo/pending file lists
                todo_file = os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bz_todo.txt')

                # Check multiple possible locations for pending files
                possible_files = [
                    todo_file,
                    os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bz_pending.txt'),
                    os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bz_filestodo.txt'),
                    os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bztodo.txt'),
                ]

                for file_path in possible_files:
                    if os.path.exists(file_path):
                        if self.debug_mode:
                            print(f"[DEBUG] Trying to read pending files from: {file_path}")
                        pending_files = self.parse_pending_files(file_path)
                        if pending_files:
                            if self.debug_mode:
                                print(f"[DEBUG] Found {len(pending_files)} files from {os.path.basename(file_path)}")
                            return pending_files

                # Priority 2: Try Backblaze CLI
                if self.debug_mode:
                    print("[DEBUG] Trying Backblaze CLI...")
                pending_files = self.get_files_via_cli()
                if pending_files:
                    if self.debug_mode:
                        print(f"[DEBUG] Found {len(pending_files)} files from CLI")
                    return pending_files

                # Priority 3: Try to find recently modified files from log (limited)
                if self.debug_mode:
                    print("[DEBUG] Trying to parse log files...")
                pending_files = self.get_files_from_logs()
                if pending_files:
                    if self.debug_mode:
                        print(f"[DEBUG] Found {len(pending_files)} files from logs")
                    return pending_files

                # Last resort: Parse database files (WARNING: may return too many files)
                # Only use this if nothing else works
                # pending_files = self.parse_bz_database()

                if self.debug_mode:
                    print("[DEBUG] No pending files found from any source")

        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Error in get_pending_files: {e}")
            pass

        return pending_files

    def get_files_via_cli(self):
        """Try to get file list using Backblaze CLI"""
        files = []
        try:
            if not self.bz_executable:
                return files

            # Try bztransmit or similar commands to list files
            # Note: Backblaze doesn't have a standard CLI for this, so this may not work
            commands = [
                f'"{self.bz_executable}" -listfiles',
                f'"{self.bz_executable}" /listfiles',
            ]

            for cmd in commands:
                stdout, stderr = self.run_command(cmd)
                if stdout:
                    # Parse output for file paths
                    for line in stdout.split('\n'):
                        line = line.strip()
                        # Look for lines that contain file paths
                        if line and ':' in line:
                            files.append({
                                'path': line,
                                'size': None,
                                'status': 'Pending'
                            })
                    if files:
                        break
        except:
            pass
        return files

    def parse_bz_database(self):
        """Parse Backblaze database files for pending file information"""
        files = []
        try:
            if not self.bz_data_path:
                return files

            # Look for bzfilelist.dat or similar database files
            db_files = [
                os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bzfilelist.dat'),
                os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bzfileids.dat'),
                os.path.join(self.bz_data_path, 'bzdata', 'bzfiledb.db'),
            ]

            for db_file in db_files:
                if os.path.exists(db_file):
                    # Try to parse as text file first
                    with open(db_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Look for file paths in the content
                        matches = re.findall(r'[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]+', content)
                        for match in matches:
                            # Filter out Backblaze's own directories
                            if ('ProgramData\\Backblaze' not in match and
                                'Program Files' not in match and
                                '\\bz' not in match.lower()):
                                files.append({
                                    'path': match,
                                    'size': None,
                                    'status': 'Pending'
                                })
                        if files:
                            break
        except:
            pass

        # Remove duplicates
        seen = set()
        unique_files = []
        for f in files:
            if f['path'] not in seen:
                seen.add(f['path'])
                unique_files.append(f)

        return unique_files

    def get_files_from_logs(self):
        """Extract file paths from Backblaze log files"""
        files = []
        try:
            if not self.bz_data_path:
                return files

            log_files = [
                os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bz_todo.txt'),
                os.path.join(self.bz_data_path, 'bzdata', 'bzbackup', 'bz_done.txt'),
                os.path.join(self.bz_data_path, 'bzlogs', 'bzbackup.log'),
                os.path.join(self.bz_data_path, 'bzdata', 'bztransmit', 'bztransmit.log'),
            ]

            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        # Read last portion of file to get recent entries (limit to last 100KB for speed)
                        f.seek(0, 2)  # Go to end
                        file_size = f.tell()
                        read_size = min(100 * 1024, file_size)  # Read last 100KB max
                        f.seek(max(0, file_size - read_size))
                        content = f.read()

                        # Look for file paths
                        matches = re.findall(r'[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]+\.\w+', content)
                        for match in matches:
                            # Filter out Backblaze's own files and directories
                            if ('ProgramData\\Backblaze' not in match and
                                'Program Files' not in match and
                                not match.endswith('.dat') and
                                not match.endswith('.log') and
                                '\\bz' not in match.lower()):
                                files.append({
                                    'path': match,
                                    'size': None,
                                    'status': 'Pending'
                                })
                                # Limit to 1000 files from logs as safety measure
                                if len(files) >= 1000:
                                    break
                    if files:
                        break
        except:
            pass

        # Remove duplicates
        seen = set()
        unique_files = []
        for f in files:
            if f['path'] not in seen:
                seen.add(f['path'])
                unique_files.append(f)

        return unique_files

    def parse_pending_files(self, file_path):
        """Parse file containing list of pending files"""
        files = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    # Filter out Backblaze's own directories
                    if line and 'ProgramData\\Backblaze' not in line and 'Program Files' not in line:
                        files.append({
                            'path': line,
                            'size': None,
                            'status': 'Pending'
                        })
        except:
            pass
        return files

    def get_mock_pending_files(self):
        """Return mock data when real data is unavailable"""
        return [
            {
                'path': 'C:\\Users\\Documents\\file1.txt',
                'size': 1024 * 500,  # 500 KB
                'status': 'Pending'
            },
            {
                'path': 'C:\\Users\\Documents\\file2.pdf',
                'size': 1024 * 1024 * 2,  # 2 MB
                'status': 'Pending'
            },
            {
                'path': 'C:\\Users\\Pictures\\photo.jpg',
                'size': 1024 * 1024 * 5,  # 5 MB
                'status': 'Pending'
            },
            {
                'path': 'C:\\Users\\Videos\\video.mp4',
                'size': 1024 * 1024 * 100,  # 100 MB
                'status': 'Pending'
            }
        ]

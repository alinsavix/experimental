using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace QuickLaunch
{
    public partial class MainWindow : Window
    {
        private List<AppEntry> applications = new List<AppEntry>();

        [DllImport("user32.dll")]
        private static extern bool GetCursorPos(out POINT lpPoint);

        [DllImport("user32.dll")]
        private static extern bool SetForegroundWindow(IntPtr hWnd);

        [StructLayout(LayoutKind.Sequential)]
        private struct POINT
        {
            public int X;
            public int Y;
        }

        public MainWindow(string[] commandLineArgs)
        {
            InitializeComponent();
            LoadApplications(commandLineArgs);

            // Position and show window after it's loaded
            this.Loaded += (s, e) =>
            {
                PositionWindowAtCursor();
                this.Activate();
                this.Focus();

                // Force window to foreground
                var helper = new System.Windows.Interop.WindowInteropHelper(this);
                SetForegroundWindow(helper.Handle);
            };
        }

        private void PositionWindowAtCursor()
        {
            // Get cursor position in physical pixels
            if (GetCursorPos(out POINT cursorPosition))
            {
                // Get DPI scaling factor
                var source = PresentationSource.FromVisual(this);
                if (source != null)
                {
                    double dpiX = source.CompositionTarget.TransformToDevice.M11;
                    double dpiY = source.CompositionTarget.TransformToDevice.M22;

                    // Convert physical pixels to WPF device-independent pixels
                    double cursorX = cursorPosition.X / dpiX;
                    double cursorY = cursorPosition.Y / dpiY;

                    // Ensure window is rendered to get actual size
                    this.UpdateLayout();
                    double windowWidth = this.ActualWidth;
                    double windowHeight = this.ActualHeight;

                    // Get screen bounds
                    double screenWidth = SystemParameters.PrimaryScreenWidth;
                    double screenHeight = SystemParameters.PrimaryScreenHeight;

                    // Calculate position, ensuring window stays on screen
                    double left = cursorX;
                    double top = cursorY;

                    // Adjust if window would go off right edge
                    if (left + windowWidth > screenWidth)
                    {
                        left = screenWidth - windowWidth;
                    }

                    // Adjust if window would go off bottom edge
                    if (top + windowHeight > screenHeight)
                    {
                        top = screenHeight - windowHeight;
                    }

                    // Ensure window doesn't go off left or top edges
                    if (left < 0) left = 0;
                    if (top < 0) top = 0;

                    this.Left = left;
                    this.Top = top;
                }
                else
                {
                    // Fallback if DPI info not available
                    this.Left = cursorPosition.X;
                    this.Top = cursorPosition.Y;
                }
            }
        }

        private void Window_Deactivated(object sender, EventArgs e)
        {
            // Close window when it loses focus (user clicks outside)
            Application.Current.Shutdown();
        }

        private void LoadApplications(string[] args)
        {
            try
            {
                applications = ConfigLoader.LoadConfig(args);
                AppListBox.ItemsSource = applications;

                if (applications.Count == 0)
                {
                    MessageBox.Show("No applications found in configuration file.",
                                    "Warning",
                                    MessageBoxButton.OK,
                                    MessageBoxImage.Warning);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading configuration: {ex.Message}",
                                "Error",
                                MessageBoxButton.OK,
                                MessageBoxImage.Error);
                Application.Current.Shutdown();
            }
        }

        private void AppListBox_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (AppListBox.SelectedItem is AppEntry selectedApp)
            {
                LaunchApplication(selectedApp);
            }
        }

        private void LaunchApplication(AppEntry app)
        {
            try
            {
                string pathToLaunch = app.Path;

                // If the path doesn't exist, try adding .lnk extension
                // (Windows Explorer hides .lnk extensions by default)
                if (!System.IO.File.Exists(pathToLaunch) && !System.IO.Directory.Exists(pathToLaunch))
                {
                    string lnkPath = pathToLaunch + ".lnk";
                    if (System.IO.File.Exists(lnkPath))
                    {
                        pathToLaunch = lnkPath;
                    }
                }

                // Start the application
                // UseShellExecute = true handles shortcuts (.lnk), executables, and other file types
                var startInfo = new ProcessStartInfo
                {
                    FileName = pathToLaunch,
                    UseShellExecute = true
                };

                Process.Start(startInfo);

                // Exit the launcher
                Application.Current.Shutdown();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error launching application: {ex.Message}",
                                "Error",
                                MessageBoxButton.OK,
                                MessageBoxImage.Error);
            }
        }
    }
}

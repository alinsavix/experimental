using System.Windows;

namespace QuickLaunch
{
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            // Pass command line arguments to MainWindow
            var mainWindow = new MainWindow(e.Args);
            mainWindow.Show();
        }
    }
}

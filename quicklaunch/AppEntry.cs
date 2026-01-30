using System;
using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace QuickLaunch
{
    /// <summary>
    /// Represents an application entry with name, path, and optional icon
    /// </summary>
    public class AppEntry
    {
        public string Name { get; set; } = string.Empty;
        public string Path { get; set; } = string.Empty;
        public string? IconPath { get; set; }

        // For WPF binding - loads the icon image
        public ImageSource? Icon
        {
            get
            {
                if (!string.IsNullOrEmpty(IconPath) && File.Exists(IconPath))
                {
                    try
                    {
                        // Check if it's an executable file
                        if (IconPath.EndsWith(".exe", StringComparison.OrdinalIgnoreCase) ||
                            IconPath.EndsWith(".dll", StringComparison.OrdinalIgnoreCase))
                        {
                            // Extract icon from executable
                            using (System.Drawing.Icon? icon = System.Drawing.Icon.ExtractAssociatedIcon(IconPath))
                            {
                                if (icon != null)
                                {
                                    return Imaging.CreateBitmapSourceFromHIcon(
                                        icon.Handle,
                                        Int32Rect.Empty,
                                        BitmapSizeOptions.FromEmptyOptions());
                                }
                            }
                        }
                        else
                        {
                            // Load as regular image file (png, jpg, ico, etc.)
                            return new BitmapImage(new Uri(IconPath, UriKind.RelativeOrAbsolute));
                        }
                    }
                    catch
                    {
                        return null;
                    }
                }
                return null;
            }
        }
    }
}

# QuickLaunch

A modern Windows application launcher built with WPF and .NET 8.

## Features

- Load application configurations from a YAML file
- Display applications in a clean, modern interface
- Appears at cursor position when launched
- Support for custom icons (optional)
- Launch applications, shortcuts (.lnk), and folders with a click
- Command-line support for custom config files
- Auto-closes when focus is lost

## Building the Application

```powershell
dotnet build
```

## Publishing the Application

To create a standalone executable:

```powershell
dotnet publish -c Release -r win-x64 --self-contained false
```

The executable will be in `bin\Release\net8.0-windows\win-x64\publish\`

## Running the Application

### Using the default config file (apps.yaml):
```powershell
dotnet run
```

### Using a custom config file:
```powershell
dotnet run -- C:\path\to\custom-config.yaml
```

Or after building:
```powershell
.\bin\Debug\net8.0-windows\QuickLaunch.exe
.\bin\Debug\net8.0-windows\QuickLaunch.exe C:\path\to\custom-config.yaml
```

## Configuration File Format

The configuration file is a YAML file with the following structure:

```yaml
applications:
  - name: Application Name
    path: C:\Path\To\Application.exe
    iconPath: C:\Path\To\Icon.png
  - name: Shortcut Name
    path: C:\Path\To\Shortcut.lnk
    iconPath: C:\Path\To\Icon.ico
```

### Fields:
- **name** (required): Display name of the application
- **path** (required): Full path to executable, shortcut (.lnk), or folder
- **iconPath** (optional): Path to an icon file (PNG, ICO, etc.)

## Sample Configuration

A sample [apps.yaml](apps.yaml) file is included with common Windows applications.

## Usage

1. Start the launcher
2. Browse the list of available applications
3. Double-click an application to launch it
4. The launcher will automatically exit after launching the selected application

## Requirements

- Windows 10 or later
- .NET 8.0 Runtime (for running the built executable)

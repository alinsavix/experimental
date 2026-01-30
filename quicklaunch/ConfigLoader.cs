using System;
using System.Collections.Generic;
using System.IO;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace QuickLaunch
{
    /// <summary>
    /// Loads application configuration from YAML file
    /// </summary>
    public class ConfigLoader
    {
        private const string DefaultConfigFile = "apps.yaml";

        public static List<AppEntry> LoadConfig(string[] commandLineArgs)
        {
            string configPath = GetConfigPath(commandLineArgs);

            if (!File.Exists(configPath))
            {
                throw new FileNotFoundException($"Configuration file not found: {configPath}");
            }

            string yaml = File.ReadAllText(configPath);
            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(CamelCaseNamingConvention.Instance)
                .Build();

            var config = deserializer.Deserialize<ConfigFile>(yaml);

            return config?.Applications ?? new List<AppEntry>();
        }

        private static string GetConfigPath(string[] args)
        {
            // Check if config file was specified on command line
            if (args.Length > 0 && !string.IsNullOrWhiteSpace(args[0]))
            {
                return args[0];
            }

            // Use default config file
            return DefaultConfigFile;
        }

        private class ConfigFile
        {
            public List<AppEntry> Applications { get; set; } = new List<AppEntry>();
        }
    }
}

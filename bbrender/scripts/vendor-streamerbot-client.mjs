import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const vendorDir = path.join(projectRoot, 'vendor');
const clientPath = path.join(vendorDir, 'streamerbot-client.js');
const metaPath = path.join(vendorDir, 'streamerbot-client.json');
const packageName = '@streamerbot/client';
const registryUrl = 'https://registry.npmjs.org/@streamerbot%2fclient';

async function main() {
  const command = process.argv[2] || 'check';
  const meta = readJson(metaPath);
  const latest = await getLatestVersion();

  if (command === 'check') {
    console.log(`Vendored ${packageName}: ${meta.version}`);
    console.log(`Latest ${packageName}: ${latest}`);
    if (meta.version === latest) {
      console.log('Vendored Streamer.bot client is current.');
      return;
    }
    console.log(`Update available: ${meta.version} -> ${latest}`);
    process.exitCode = 1;
    return;
  }

  if (command === 'update') {
    await updateVendor(latest);
    return;
  }

  throw new Error(`Unknown command: ${command}`);
}

async function getLatestVersion() {
  const response = await fetch(registryUrl);
  if (!response.ok) {
    throw new Error(`Failed to read npm registry metadata: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  const latest = data?.['dist-tags']?.latest;
  if (!latest) throw new Error(`Could not find latest dist-tag for ${packageName}`);
  return latest;
}

async function updateVendor(version) {
  const source = `https://cdn.jsdelivr.net/npm/${packageName}@${version}/dist/streamerbot-client.js`;
  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`Failed to download ${source}: ${response.status} ${response.statusText}`);
  }

  const clientSource = await response.text();
  mkdirSync(vendorDir, { recursive: true });
  writeFileSync(clientPath, clientSource, 'utf8');
  writeFileSync(metaPath, JSON.stringify({
    package: packageName,
    version,
    source,
    vendoredAt: new Date().toISOString(),
    sha256: sha256Hex(clientSource)
  }, null, 2) + '\n', 'utf8');

  console.log(`Updated vendored ${packageName} to ${version}`);
}

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

function sha256Hex(value) {
  return createHash('sha256').update(value).digest('hex');
}

main().catch((error) => {
  console.error(error.message || error);
  process.exitCode = 1;
});

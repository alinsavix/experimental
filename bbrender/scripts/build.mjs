import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const distRoot = path.join(projectRoot, 'dist');
const adapterHtmlFile = 'sb_bbrender.html';
const sourceHtmlPath = path.join(projectRoot, adapterHtmlFile);
const packageJsonPath = path.join(projectRoot, 'package.json');
const vendorClientPath = path.join(projectRoot, 'vendor', 'streamerbot-client.js');
const vendorClientMetaPath = path.join(projectRoot, 'vendor', 'streamerbot-client.json');

const moduleOrder = [
  'src/utils.js',
  'src/tag-styles.js',
  'src/tag-registry.js',
  'src/tag-definitions.js',
  'src/parser.js',
  'src/base-styles.js',
  'src/dom-renderer.js',
  'src/layout.js',
  'src/runtime-effects.js',
  'src/payload.js',
  'src/transitions.js',
  'src/index.js'
];

const publicApi = [
  'parse',
  'parseAttributes',
  'parseTagToken',
  'renderToFragment',
  'applySourceLayout',
  'extractBBCode',
  'getCommand',
  'getPayloadData',
  'shouldHandlePayload',
  'normalizeColor',
  'applyAnimationStyles',
  'applyRootTransition',
  'clearRootTransition',
  'normalizeTransition',
  'splitGraphemes',
  'knownTags',
  'renderToElement',
  'wrapWithGlobalTags',
  'createState'
];

export function build() {
  const packageInfo = JSON.parse(readFile(packageJsonPath));
  const vendorClientMeta = JSON.parse(readFile(vendorClientMetaPath));
  const vendorClientSource = readFile(vendorClientPath);
  const sourceHtml = readFile(sourceHtmlPath);
  const css = extractBlock(sourceHtml, /<style>\s*([\s\S]*?)\s*<\/style>/i, 'style');
  const overlaySource = extractBlock(
    sourceHtml,
    /<script\s+type="module">\s*([\s\S]*?)\s*<\/script>/i,
    'module script'
  ).replace(/^\s*import\s+\*\s+as\s+BBRender\s+from\s+['"]\.\/bbrender\.js['"];\s*/m, '');

  const bundledJs = [
    '// vendored Streamer.bot client',
    vendorClientSource.trim(),
    '',
    '"use strict";',
    '(() => {',
    buildRendererBundle(),
    '',
    '// bbrender overlay',
    overlaySource.trim(),
    '})();',
    ''
  ].join('\n');
  const cssText = `${css.trim()}\n`;

  rmSync(distRoot, { recursive: true, force: true });
  const packedDir = path.join(distRoot, 'packed');
  const separateDir = path.join(distRoot, 'separate');
  const cdnDir = path.join(distRoot, 'cdn');
  const cdnAssetsDir = path.join(cdnDir, 'assets');
  mkdirSync(packedDir, { recursive: true });
  mkdirSync(separateDir, { recursive: true });
  mkdirSync(cdnAssetsDir, { recursive: true });

  const buildHtml = stripVendorClientScript(sourceHtml);

  const separateHtml = buildHtml
    .replace(/<style>[\s\S]*?<\/style>/i, '<link rel="stylesheet" href="./bbrender.css">')
    .replace(/<script\s+type="module">[\s\S]*?<\/script>/i, '<script src="./bbrender.js"></script>');

  const packedHtml = buildHtml
    .replace(/<style>[\s\S]*?<\/style>/i, `<style>\n${css.trim()}\n  </style>`)
    .replace(/<script\s+type="module">[\s\S]*?<\/script>/i, `<script>\n${bundledJs}</script>`);

  writeFileSync(path.join(separateDir, adapterHtmlFile), separateHtml, 'utf8');
  writeFileSync(path.join(separateDir, 'bbrender.css'), cssText, 'utf8');
  writeFileSync(path.join(separateDir, 'bbrender.js'), bundledJs, 'utf8');
  writeFileSync(path.join(packedDir, adapterHtmlFile), packedHtml, 'utf8');

  const cssHash = shortHash(cssText);
  const jsHash = shortHash(bundledJs);
  const cdnCssFile = `bbrender.${cssHash}.css`;
  const cdnJsFile = `bbrender.${jsHash}.js`;
  const cdnHtml = buildHtml
    .replace(/<style>[\s\S]*?<\/style>/i, `<link rel="stylesheet" href="./assets/${cdnCssFile}">`)
    .replace(/<script\s+type="module">[\s\S]*?<\/script>/i, `<script src="./assets/${cdnJsFile}"></script>`);

  writeFileSync(path.join(cdnDir, adapterHtmlFile), cdnHtml, 'utf8');
  writeFileSync(path.join(cdnAssetsDir, cdnCssFile), cssText, 'utf8');
  writeFileSync(path.join(cdnAssetsDir, cdnJsFile), bundledJs, 'utf8');
  writeFileSync(path.join(cdnDir, 'manifest.json'), JSON.stringify({
    name: packageInfo.name || 'bbrender',
    version: packageInfo.version || '0.0.0',
    generatedAt: new Date().toISOString(),
    entrypoint: adapterHtmlFile,
    assets: {
      css: `assets/${cdnCssFile}`,
      js: `assets/${cdnJsFile}`
    },
    vendor: {
      streamerbotClient: {
        package: vendorClientMeta.package,
        version: vendorClientMeta.version,
        source: vendorClientMeta.source,
        sha256: sha256Hex(vendorClientSource),
        sri: sri(vendorClientSource)
      }
    },
    hashes: {
      css: {
        sha256: sha256Hex(cssText),
        sri: sri(cssText)
      },
      js: {
        sha256: sha256Hex(bundledJs),
        sri: sri(bundledJs)
      }
    },
    cache: {
      html: 'short-lived or revalidated',
      assets: 'immutable; safe for long-lived CDN caching'
    }
  }, null, 2) + '\n', 'utf8');

  return {
    packed: path.join(packedDir, adapterHtmlFile),
    separate: separateDir,
    cdn: cdnDir
  };
}

function buildRendererBundle() {
  const moduleSources = moduleOrder.map((modulePath) => {
    const source = readFile(path.join(projectRoot, modulePath));
    return [`// ${modulePath}`, stripModuleSyntax(source).trim()].join('\n');
  });

  return [
    '// bbrender renderer',
    ...moduleSources,
    '',
    `const BBRender = Object.freeze({ ${publicApi.join(', ')} });`,
    "if (typeof window !== 'undefined') window.BBRender = BBRender;"
  ].join('\n\n');
}

function stripModuleSyntax(source) {
  return source
    .replace(/^\s*import\s+[^;]+;\s*$/gm, '')
    .replace(/^\s*export\s+\{[\s\S]*?\}\s+from\s+['"][^'"]+['"];\s*/gm, '')
    .replace(/^\s*export\s+\{[\s\S]*?\};\s*/gm, '')
    .replace(/\bexport\s+(function|const|let|var|class)\s+/g, '$1 ');
}

function extractBlock(source, pattern, label) {
  const match = source.match(pattern);
  if (!match) throw new Error(`Could not find ${label} block in ${adapterHtmlFile}`);
  return match[1];
}

function stripVendorClientScript(source) {
  return source.replace(/^\s*<script\s+src=["']\.\/vendor\/streamerbot-client\.js["']><\/script>\s*$/im, '');
}

function readFile(filePath) {
  if (!existsSync(filePath)) throw new Error(`Missing file: ${filePath}`);
  return readFileSync(filePath, 'utf8');
}

function sha256Hex(value) {
  return createHash('sha256').update(value).digest('hex');
}

function shortHash(value) {
  return sha256Hex(value).slice(0, 12);
}

function sri(value) {
  return `sha256-${createHash('sha256').update(value).digest('base64')}`;
}

const isMain = process.argv[1] && normalizedScriptPath(process.argv[1]).endsWith('/scripts/build.mjs');

if (isMain) {
  const result = build();
  console.log(`Built packed artifact: ${path.relative(projectRoot, result.packed)}`);
  console.log(`Built separate artifacts: ${path.relative(projectRoot, result.separate)}`);
  console.log(`Built CDN artifacts: ${path.relative(projectRoot, result.cdn)}`);
}

function normalizedScriptPath(value) {
  return String(value || '').replace(/\\/g, '/').toLowerCase();
}

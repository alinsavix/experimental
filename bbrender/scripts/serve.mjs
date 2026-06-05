import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const rootArg = process.argv[2] || '.';
const portArg = process.argv[3] || process.env.PORT || '4173';
const root = path.resolve(projectRoot, rootArg);
const port = Number.parseInt(portArg, 10) || 4173;

const contentTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8'
};

if (!existsSync(root) || !statSync(root).isDirectory()) {
  console.error(`Serve root does not exist or is not a directory: ${root}`);
  process.exit(1);
}

const server = createServer((request, response) => {
  const url = new URL(request.url || '/', `http://${request.headers.host || 'localhost'}`);
  const decodedPath = decodeURIComponent(url.pathname);
  const requestedPath = path.resolve(root, `.${decodedPath}`);

  if (!isInside(root, requestedPath)) {
    response.writeHead(403);
    response.end('Forbidden');
    return;
  }

  let filePath = requestedPath;
  if (existsSync(filePath) && statSync(filePath).isDirectory()) {
    filePath = path.join(filePath, 'sb_bbrender.html');
  }

  if (!existsSync(filePath) || !statSync(filePath).isFile()) {
    response.writeHead(404);
    response.end('Not found');
    return;
  }

  response.writeHead(200, {
    'Content-Type': contentTypes[path.extname(filePath).toLowerCase()] || 'application/octet-stream'
  });
  createReadStream(filePath).pipe(response);
});

server.listen(port, '127.0.0.1', () => {
  console.log(`Serving ${root}`);
  console.log(`http://127.0.0.1:${port}/`);
});

function isInside(parent, child) {
  const relative = path.relative(parent, child);
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
}

import { build } from './build.mjs';
import { runTests } from './test.mjs';

runTests();

const result = build();
console.log(`Built packed artifact: ${result.packed}`);
console.log(`Built separate artifacts: ${result.separate}`);
console.log(`Built CDN artifacts: ${result.cdn}`);

import { expect, test } from '@playwright/test';

const staticVisualCaseIds = [
  'plain',
  'unknown',
  'noparse',
  'global-tags',
  'formatting-static',
  'colors-static',
  'gradient-static',
  'styling-static',
  'stroke-static',
  'box-hr',
  'dropcap',
  'dropcap-lines',
  'wrap-word',
  'wrap-char',
  'inline-image',
  'payload-command'
];

test('browser renderer checks pass', async ({ page }) => {
  await page.goto('/tests/visual-harness.html?mode=tests');

  const summaryHandle = await page.waitForFunction(() => window.__BBCodeTestResults);
  const summary = await summaryHandle.jsonValue();

  expect(summary.failed).toBe(0);
  expect(summary.passed).toBeGreaterThan(0);
});

for (const caseId of staticVisualCaseIds) {
  test(`static visual case: ${caseId}`, async ({ page }) => {
    await page.goto(`/tests/visual-harness.html?case=${encodeURIComponent(caseId)}`);
    await disableMotion(page);

    const caseShell = page.locator(`[data-case-id="${caseId}"]`);
    const preview = caseShell.locator('.preview');

    await expect(page.locator('.case')).toHaveCount(1);
    await expect(preview).toBeVisible();
    await expect(preview).toHaveScreenshot(`${caseId}.png`);
  });
}

async function disableMotion(page) {
  await page.addStyleTag({
    content: `
      *,
      *::before,
      *::after {
        animation: none !important;
        transition: none !important;
        caret-color: transparent !important;
      }
    `
  });
}

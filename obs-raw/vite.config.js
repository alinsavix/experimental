import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vuetify from 'vite-plugin-vuetify'

export default defineConfig({
  base: './',
  plugins: [
    vue(),
    vuetify({ autoImport: true }),
    {
      name: 'woff2-only',
      enforce: 'pre',
      transform(code, id) {
        if (id.includes('@fontsource') && id.endsWith('.css')) {
          return code.replace(/,\s*url\([^)]+\.woff\)\s*format\('woff'\)/g, '')
        }
      },
      generateBundle(_, bundle) {
        for (const key of Object.keys(bundle)) {
          if (key.endsWith('.woff')) delete bundle[key]
        }
      },
    },
  ],
})

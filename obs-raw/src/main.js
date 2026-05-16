import { createApp } from 'vue'
import App from './App.vue'
import vuetify from './plugins/vuetify'
import WebFont from 'webfontloader'

WebFont.load({
  google: { families: ['Poppins:100,300,400,500,700,900&display=swap'] },
})

createApp(App).use(vuetify).mount('#app')

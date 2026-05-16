<template>
  <div class="position-relative">
    <pre
      class="pa-3 overflow-y-auto rounded-lg bg-grey-darken-4"
      style="min-height: 4rem; max-height: 500px"
    ><code class="language-javascript">{{ displayValue }}</code></pre>
    <span class="position-absolute" style="top: 1rem; right: 1.5rem">
      <v-tooltip location="top">
        <template #activator="{ props: tooltipProps }">
          <span v-bind="tooltipProps">
            <v-btn size="x-small" icon :disabled="!isSupported" @click="copy(displayValue)">
              <v-icon size="small">mdi-content-copy</v-icon>
            </v-btn>
          </span>
        </template>
        <small class="text-white">{{ isSupported ? 'Copy' : 'Clipboard Permission Required' }}</small>
      </v-tooltip>
    </span>
  </div>
</template>

<script setup>
import { computed, onMounted, watch, nextTick } from 'vue'
import { useClipboard } from '@vueuse/core'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'

const props = defineProps({
  value: null,
})

const { copy, isSupported } = useClipboard()

const displayValue = computed(() =>
  typeof props.value === 'string' ? props.value : JSON.stringify(props.value, null, 2),
)

function highlight() {
  nextTick(() => Prism.highlightAll())
}

onMounted(highlight)
watch(() => props.value, highlight)
</script>

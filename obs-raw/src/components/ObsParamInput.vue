<template>
  <!-- Boolean param: True/False select -->
  <v-select
    v-if="param.type === 'Boolean'"
    v-model="modelValue"
    :items="[{ value: true, title: 'True' }, { value: false, title: 'False' }]"
    label="Value"
    density="compact"
    :hint="param.description"
    persistent-hint
    class="mr-2 mb-2"
  />

  <!-- Object param: JSON textarea -->
  <v-textarea
    v-else-if="param.type === 'Object'"
    v-model="modelValue"
    label="JSON Value"
    density="compact"
    :hint="param.description"
    persistent-hint
    class="mr-2 mb-2"
    validate-on-blur
    auto-grow
    :rules="[isValidJson(modelValue) || 'Invalid JSON']"
  />

  <!-- String/Number with dynamic options: select -->
  <v-select
    v-else-if="options.length"
    v-model="modelValue"
    :items="options"
    label="Value"
    density="compact"
    :hint="param.description"
    persistent-hint
    class="mr-2 mb-2"
  />

  <!-- String/Number: plain text field -->
  <v-text-field
    v-else
    v-model="modelValue"
    label="Value"
    density="compact"
    :hint="param.description"
    persistent-hint
    class="mr-2 mb-2"
  />
</template>

<script setup>
import { ref, computed, watch, onMounted } from 'vue'

const props = defineProps({
  param: { type: Object, required: true },
  value: { default: null },
  obs: { default: null },
  ctx: { default: null },
})

const emit = defineEmits(['update:value'])

const modelValue = computed({
  get: () => props.value,
  set: (val) => emit('update:value', val),
})

const options = ref([])

function isValidJson(str) {
  try {
    JSON.parse(str)
    return true
  } catch {
    return false
  }
}

async function fetchOptions() {
  if (props.param.getOptions && props.obs?.identified) {
    try {
      const result = await props.param.getOptions(props.obs, props.ctx)
      if (typeof result[0] === 'object') {
        // Array of { title, value } objects
        if (modelValue.value && !result.find((r) => r.value === modelValue.value)) {
          modelValue.value = null
        }
        options.value = result
      } else {
        // Array of strings
        if (modelValue.value && !result.includes(modelValue.value)) {
          modelValue.value = null
        }
        options.value = result
      }
    } catch (e) {
      console.error(e)
    }
  } else if (props.param.getValue && props.obs?.identified) {
    try {
      const result = await props.param.getValue(props.obs, props.ctx)
      if (result != null) {
        modelValue.value = JSON.stringify(result)
      }
    } catch {
      // ignore
    }
  }
}

onMounted(fetchOptions)

watch(
  [() => props.obs?.identified, () => props.param, () => props.ctx],
  fetchOptions,
)
</script>

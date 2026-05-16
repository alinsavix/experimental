<template>
  <v-sheet class="rounded-lg">
    <v-text-field
      v-model="searchQuery"
      label="Search"
      density="compact"
      variant="solo"
      hide-details
      clearable
    />
    <v-list color="primary" nav class="request-list overflow-y-auto">
      <template v-for="(request, index) in filteredRequests" :key="request.name">
        <v-list-subheader
          v-if="request.group !== filteredRequests[index - 1]?.group"
          class="bg-grey-darken-4 mt-3"
        >
          <span class="text-primary text-uppercase font-weight-bold">{{ request.group }}</span>
        </v-list-subheader>
        <v-list-item
          :active="modelValue?.name === request.name"
          :disabled="obsInfo && !obsInfo.availableRequests?.includes(request.name)"
          color="primary"
          :href="`#${request.name}`"
          @click="modelValue = request"
        >
          <v-list-item-title>{{ request.name }}</v-list-item-title>
          <v-list-item-subtitle>{{ request.description }}</v-list-item-subtitle>
        </v-list-item>
      </template>
    </v-list>
  </v-sheet>
</template>

<script setup>
import { ref, computed } from 'vue'
import { refDebounced } from '@vueuse/core'
import { requests } from '../data/requests.js'

const props = defineProps({
  modelValue: { default: null },
  obsInfo: { default: null },
})

const emit = defineEmits(['update:modelValue'])

const modelValue = computed({
  get: () => props.modelValue,
  set: (val) => emit('update:modelValue', val),
})

const searchQuery = ref('')
const debouncedQuery = refDebounced(searchQuery, 500)

const filteredRequests = computed(() => {
  if (!debouncedQuery.value) return requests
  const re = new RegExp(debouncedQuery.value, 'i')
  return requests.filter((r) => `${r.name} ${r.description} ${r.group}`.match(re))
})
</script>

<style scoped>
.request-list {
  max-height: calc(100vh - 120px);
}
</style>

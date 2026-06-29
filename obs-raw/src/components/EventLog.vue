<template>
  <div>
    <div class="d-flex flex-wrap align-center ga-3 mb-4">
      <div>
        <h2>OBS Events</h2>
        <div class="text-caption text-medium-emphasis">
          Newest first · keeping the latest {{ maxEvents }} events
        </div>
      </div>

      <v-spacer />

      <v-chip :color="isConnected ? 'success' : 'default'" variant="tonal">
        <v-icon start size="small">
          {{ isConnected ? 'mdi-check-circle' : 'mdi-lan-disconnect' }}
        </v-icon>
        {{ isConnected ? 'Listening' : 'Disconnected' }}
      </v-chip>

      <v-btn
        variant="outlined"
        color="error"
        :disabled="events.length === 0"
        @click="$emit('clear')"
      >
        Clear
        <v-icon end>mdi-delete-outline</v-icon>
      </v-btn>
    </div>

    <v-sheet class="rounded-lg pa-3 mb-4">
      <v-select
        v-model="selectedTypes"
        :items="availableTypes"
        label="Filter by event type"
        multiple
        chips
        closable-chips
        clearable
        hide-details
        :disabled="availableTypes.length === 0"
      />
      <div class="text-caption text-medium-emphasis mt-2">
        Showing {{ filteredEvents.length }} of {{ events.length }} received events.
        High-volume event categories are excluded by OBS's default subscription.
      </div>
    </v-sheet>

    <v-alert v-if="!isConnected && events.length === 0" type="info" variant="tonal">
      Connect to OBS to start receiving events.
    </v-alert>

    <v-alert
      v-else-if="events.length > 0 && filteredEvents.length === 0"
      type="info"
      variant="tonal"
    >
      No received events match the selected types.
    </v-alert>

    <v-alert v-else-if="events.length === 0" type="info" variant="tonal">
      Listening for OBS events. Changes made in OBS will appear here.
    </v-alert>

    <v-expansion-panels v-else multiple variant="accordion">
      <v-expansion-panel v-for="event in summarizedEvents" :key="event.id">
        <v-expansion-panel-title>
          <div class="event-heading">
            <span class="event-identity">
              <span class="font-weight-medium text-primary">{{ event.type }}</span>
              <span v-if="event.summary" class="event-summary">
                {{ event.summary }}
              </span>
            </span>
            <time class="text-caption text-medium-emphasis" :datetime="event.receivedAt">
              {{ formatTime(event.receivedAt) }}
            </time>
          </div>
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <pre class="event-data pa-3 rounded-lg bg-grey-darken-4"><code>{{ formatData(event.data) }}</code></pre>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { summarizeEvent } from '../domain/eventSummary.mjs'

const props = defineProps({
  events: { type: Array, default: () => [] },
  isConnected: Boolean,
  maxEvents: { type: Number, default: 500 },
})

defineEmits(['clear'])

const selectedTypes = ref([])

const availableTypes = computed(() =>
  [...new Set(props.events.map((event) => event.type))].sort((a, b) => a.localeCompare(b)),
)

const filteredEvents = computed(() => {
  if (selectedTypes.value.length === 0) return props.events
  const selected = new Set(selectedTypes.value)
  return props.events.filter((event) => selected.has(event.type))
})

const summarizedEvents = computed(() =>
  filteredEvents.value.map((event) => ({
    ...event,
    summary: summarizeEvent(event.type, event.data),
  })),
)

function formatTime(timestamp) {
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    fractionalSecondDigits: 3,
  }).format(new Date(timestamp))
}

function formatData(data) {
  return JSON.stringify(data ?? {}, null, 2)
}
</script>

<style scoped>
.event-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  width: 100%;
  padding-right: 1rem;
}

.event-identity {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.event-summary {
  color: rgba(var(--v-theme-on-surface), var(--v-medium-emphasis-opacity));
  font-size: 0.8rem;
  overflow-wrap: anywhere;
}

.event-heading time {
  flex-shrink: 0;
}

.event-data {
  overflow: auto;
  max-height: 28rem;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
</style>

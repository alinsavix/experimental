<template>
  <!-- Toolbar -->
  <v-toolbar density="compact" color="#1d1d1d">
    <span class="d-flex flex-column mx-5">
      <span>OBS Raw Generator</span>
      <small class="text-grey">...but actually maintained</small>
    </span>

    <v-btn-toggle v-model="currentPage" mandatory density="compact" color="primary">
      <v-btn value="requests" @click="showRequests">
        Requests
        <v-icon end>mdi-code-json</v-icon>
      </v-btn>
      <v-btn value="events" @click="showEvents">
        Events
        <v-badge v-if="events.length" :content="events.length" inline color="primary" />
        <v-icon end>mdi-format-list-bulleted</v-icon>
      </v-btn>
    </v-btn-toggle>

    <v-spacer />

    <v-toolbar-items>
      <span class="d-flex align-center mx-5">
        <v-text-field
          v-model="host"
          type="text"
          label="Host"
          density="compact"
          variant="filled"
          class="mr-2 my-1"
          hide-details
          :disabled="isConnected"
          style="min-width: 150px"
        />
        <v-text-field
          v-model.number="port"
          type="number"
          label="Port"
          density="compact"
          variant="filled"
          class="mr-2 my-1"
          hide-details
          :disabled="isConnected"
          style="min-width: 110px"
        />
        <v-text-field
          v-model="password"
          type="password"
          label="Password"
          density="compact"
          variant="filled"
          class="mr-2 my-1"
          hide-details
          :disabled="isConnected"
          style="min-width: 150px"
        />

        <v-btn
          v-if="isConnected"
          color="pink"
          variant="outlined"
          class="ml-2"
          @click="disconnect"
        >
          Disconnect
          <v-icon class="ml-2">mdi-close-circle</v-icon>
        </v-btn>
        <v-btn
          v-else
          color="success"
          variant="outlined"
          class="ml-2"
          :loading="isConnecting"
          @click="connect"
        >
          Connect
          <v-icon class="ml-2">mdi-chevron-right-circle</v-icon>
        </v-btn>
      </span>
    </v-toolbar-items>
  </v-toolbar>

  <!-- Main content -->
  <v-container fluid>
    <v-row v-if="currentPage === 'requests'">
      <!-- Left: request list -->
      <v-col cols="12" sm="4" xl="3">
        <RequestMenu v-model="selectedRequest" :obs-info="obsInfo" />
      </v-col>

      <!-- Right: request builder -->
      <v-col cols="12" sm="8" xl="9">
        <div>
          <!-- Error alert -->
          <v-slide-y-transition>
            <v-alert
              v-if="error"
              type="error"
              density="compact"
              variant="tonal"
              class="my-3"
            >
              <v-alert-title>{{ error }}</v-alert-title>
              <small>OBS Studio must be running with WebSocket enabled</small>
            </v-alert>
          </v-slide-y-transition>

          <!-- OBS version info bar -->
          <v-slide-y-transition>
            <v-sheet
              v-if="obsInfo"
              class="rounded-lg pa-3 text-grey d-flex align-center"
            >
              <span class="mr-5 text-success d-flex align-center">
                <v-icon size="small" color="success">mdi-check-circle</v-icon>
                <span class="ml-1">Connected</span>
              </span>
              <small> OBS Studio {{ obsInfo.obsVersion }}</small>
              <span class="mx-2" />
              <small> OBS WebSocket {{ obsInfo.obsWebSocketVersion }}</small>
              <span class="mx-2" />
              <small> RPC v{{ obsInfo.rpcVersion }}</small>
              <span class="mx-2" />
              <small>{{ obsInfo.platformDescription }}</small>
            </v-sheet>
          </v-slide-y-transition>

          <!-- Request builder -->
          <div class="mt-5">
            <h2>Request</h2>
            <form>
              <!-- Preview of generated request object -->
              <CodePreview v-if="selectedRequest" :value="preview" />

              <!-- Param list -->
              <v-slide-y-transition group>
                <template v-if="selectedRequest">
                  <v-sheet
                    v-for="(param, index) in displayParams"
                    :key="index"
                    class="rounded-lg my-3 pa-2"
                  >
                    <!-- Param header -->
                    <div class="d-flex justify-space-between">
                      <span class="px-2 pt-2 text-grey">Param {{ index + 1 }}</span>
                      <v-btn
                        icon
                        variant="plain"
                        size="small"
                        class="ml-2"
                        color="error"
                        :disabled="param.required"
                        @click="removeParam(param)"
                      >
                        <v-icon size="small">mdi-close-circle</v-icon>
                      </v-btn>
                    </div>

                    <!-- Param inputs -->
                    <div class="d-flex align-center">
                      <v-select
                        v-model="param.type"
                        :items="paramTypeItems"
                        label="Type"
                        density="compact"
                        :disabled="param.required"
                        class="mr-2 align-self-start"
                        style="max-width: 200px"
                      />
                      <v-text-field
                        v-model="param.name"
                        label="Name"
                        density="compact"
                        :disabled="param.required"
                        class="mr-2 align-self-start"
                        style="max-width: 350px"
                      />

                      <!-- Nested object params (e.g. keyModifiers) -->
                      <div
                        v-if="param.type === 'Object' && param.params"
                        class="d-flex flex-column"
                        style="width: 50%"
                      >
                        <ObsParamInput
                          v-for="subParam in param.params"
                          :key="`${param.name}-${subParam.name}`"
                          v-model:value="subParam.value"
                          :param="subParam"
                        />
                      </div>

                      <!-- Single value input -->
                      <div v-else style="width: 50%">
                        <ObsParamInput
                          v-model:value="param.value"
                          :param="param"
                          :obs="obs"
                          :ctx="{ ...selectedRequest, params: currentParams }"
                        />
                      </div>
                    </div>
                  </v-sheet>
                </template>
              </v-slide-y-transition>
            </form>

            <!-- Action buttons -->
            <div class="d-flex align-center">
              <v-btn
                v-if="selectedRequest"
                type="button"
                class="my-2 mr-2"
                :disabled="!selectedRequest"
                @click="addParam"
              >
                Add Param
                <v-icon class="ml-2">mdi-plus</v-icon>
              </v-btn>

              <v-spacer class="ml-auto" />

              <v-btn
                class="my-2 mr-2"
                color="primary"
                variant="text"
                :disabled="!clipboardSupported"
                @click="copyCph"
              >
                Copy CPH
                <v-icon class="ml-2">mdi-content-copy</v-icon>
              </v-btn>

              <v-btn
                class="my-2 mr-2"
                color="primary"
                variant="text"
                :disabled="!clipboardSupported"
                @click="copyObsRaw"
              >
                Copy OBS Raw
                <v-icon class="ml-2">mdi-content-copy</v-icon>
              </v-btn>

              <v-btn
                class="my-2"
                color="success"
                :disabled="!isConnected || !selectedRequest"
                @click="sendRequest"
              >
                Send Request
                <v-icon class="ml-2">mdi-send</v-icon>
              </v-btn>
            </div>
          </div>

          <!-- Response -->
          <div class="my-3">
            <h2>Response</h2>
            <v-sheet
              v-if="screenshotImageSrc"
              class="rounded-lg my-3 pa-3 d-flex justify-center bg-grey-darken-4"
            >
              <img
                :src="screenshotImageSrc"
                alt="GetSourceScreenshot response"
                style="display: block; max-width: 100%; max-height: 70vh; object-fit: contain"
              >
            </v-sheet>
            <CodePreview :value="responsePreview" />
          </div>
        </div>
      </v-col>
    </v-row>

    <v-row v-else>
      <v-col cols="12">
        <v-slide-y-transition>
          <v-alert
            v-if="error"
            type="error"
            density="compact"
            variant="tonal"
            class="mb-4"
          >
            <v-alert-title>{{ error }}</v-alert-title>
            <small>OBS Studio must be running with WebSocket enabled</small>
          </v-alert>
        </v-slide-y-transition>

        <EventLog
          :events="events"
          :is-connected="isConnected"
          :max-events="MAX_EVENTS"
          @clear="clearEvents"
        />
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useLocalStorage, useClipboard } from '@vueuse/core'
import OBSWebSocket, { EventSubscription } from 'obs-websocket-js'
import RequestMenu from './RequestMenu.vue'
import ObsParamInput from './ObsParamInput.vue'
import CodePreview from './CodePreview.vue'
import EventLog from './EventLog.vue'
import { requests, paramTypeItems, ParamType } from '../data/requests.js'
import { eventTypes } from '../data/eventTypes.js'
import { buildRequestData } from '../domain/requestData.mjs'

// OBS WebSocket instance (not reactive — passed by reference to child components)
const obs = new OBSWebSocket()

// Persisted connection settings
const host = useLocalStorage('obs-store:host', '127.0.0.1')
const port = useLocalStorage('obs-store:port', 4455)
const password = useLocalStorage('obs-store:password', '')

// Connection state
const isConnecting = ref(false)
const isConnected = ref(false)
const error = ref(null)
const obsInfo = ref(null)

// Page state + bounded in-memory event log
const currentPage = ref('requests')
const events = ref([])
const MAX_EVENTS = 500
let nextEventId = 1
const eventListeners = new Map()

// Selected request + per-request param state
const selectedRequest = ref(requests[0])
const paramState = ref({})  // { [requestName]: ParamArray }
const response = ref(null)

// Clipboard
const { copy, isSupported: clipboardSupported } = useClipboard()

// WebSocket URL
const wsUrl = computed(() => `ws://${host.value}:${port.value}`)

// Params for the currently selected request
const currentParams = computed(() => {
  if (!selectedRequest.value) return []
  return paramState.value[selectedRequest.value.name] ?? []
})

// Params actually shown/sent: version-gated params (e.g. canvasUuid) are hidden
// when connected to an OBS that lacks the required request. While disconnected
// we can't tell, so everything is shown.
const displayParams = computed(() =>
  currentParams.value.filter((p) => {
    if (!p.requiresRequest || !obsInfo.value) return true
    return obsInfo.value.availableRequests?.includes(p.requiresRequest) ?? false
  }),
)

const imageMimeTypes = {
  bmp: 'image/bmp',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  png: 'image/png',
  webp: 'image/webp',
}

const screenshotImageFormat = computed(() => {
  const format = currentParams.value.find((p) => p.name === 'imageFormat')?.value
  return String(format || 'png').toLowerCase()
})

const screenshotImageSrc = computed(() => {
  if (selectedRequest.value?.name !== 'GetSourceScreenshot') return null
  const imageData = response.value?.imageData
  if (typeof imageData !== 'string' || !imageData.trim()) return null
  const trimmed = imageData.trim()
  if (trimmed.startsWith('data:image/')) return trimmed
  const mimeType = imageMimeTypes[screenshotImageFormat.value] ?? `image/${screenshotImageFormat.value}`
  return `data:${mimeType};base64,${trimmed}`
})

const responsePreview = computed(() => {
  if (!screenshotImageSrc.value || typeof response.value?.imageData !== 'string') {
    return response.value
  }
  return {
    ...response.value,
    imageData: `[base64 image data omitted: ${response.value.imageData.length.toLocaleString()} chars]`,
  }
})

// Preview of the generated Streamer.bot OBS Raw request
const preview = computed(() => {
  return buildRequestPayload('streamerbot_raw', selectedRequest.value?.name ?? '', buildData())
})

// -------------------------------------------------------------------
// Request payload formatters
// -------------------------------------------------------------------
function buildRequestPayload(format, requestType = '', data = {}) {
  switch (format) {
    case 'obs':
      return { 'request-type': requestType, ...data }
    case 'streamerbot_raw_cph':
      return `CPH.ObsSendRaw("${requestType}", ${JSON.stringify(JSON.stringify(data))}, 0);`
    case 'streamerbot_raw':
    default:
      return { requestType, requestData: data }
  }
}

function buildData() {
  if (!selectedRequest.value) return {}
  return buildRequestData(displayParams.value)
}

// -------------------------------------------------------------------
// Watch selected request → initialise its param state
// -------------------------------------------------------------------
watch(selectedRequest, (next, prev) => {
  if (!next) return

  // Clear response when switching requests
  if (next?.name !== prev?.name) {
    response.value = null
  }

  const defined = (next.params ?? []).map((p) => ({ ...p, value: null }))

  if (!paramState.value[next.name] || !Array.isArray(paramState.value[next.name])) {
    paramState.value = { ...paramState.value, [next.name]: defined }
    return
  }

  // Merge any newly-added params from the definition into existing state
  for (const p of defined) {
    if (!paramState.value[next.name].find((existing) => existing.name === p.name)) {
      paramState.value[next.name].push({ ...p, value: null })
    }
  }
})

// Fetch OBS version info when connected
watch(isConnected, async (connected) => {
  if (!connected) {
    obsInfo.value = null
    return
  }
  try {
    obsInfo.value = await obs.call('GetVersion')
    console.log(obsInfo.value)
  } catch (e) {
    console.error(e)
  }
})

// OBS only sends event traffic while the event log is visible.
watch(currentPage, async (page) => {
  if (!isConnected.value) return
  try {
    await obs.reidentify({ eventSubscriptions: eventSubscriptionForPage(page) })
  } catch (e) {
    error.value = e.message || 'Unable to update OBS event subscriptions'
  }
})

// -------------------------------------------------------------------
// Connection
// -------------------------------------------------------------------
function setupListeners() {
  obs.on('ConnectionClosed', () => (isConnected.value = false))
  obs.on('ConnectionError', () => (isConnected.value = false))

  for (const eventType of eventTypes) {
    const listener = (data) => {
      // Also guard locally in case an event was already in flight when the
      // subscription was disabled.
      if (currentPage.value !== 'events') return

      events.value.unshift({
        id: nextEventId++,
        type: eventType,
        receivedAt: new Date().toISOString(),
        data,
      })

      if (events.value.length > MAX_EVENTS) {
        events.value.length = MAX_EVENTS
      }
    }
    eventListeners.set(eventType, listener)
    obs.on(eventType, listener)
  }
}

function eventSubscriptionForPage(page) {
  return page === 'events' ? EventSubscription.All : EventSubscription.None
}

function clearEvents() {
  events.value = []
}

function showEvents() {
  window.location.hash = 'events'
}

function showRequests() {
  window.location.hash = selectedRequest.value?.name ?? ''
}

function handleHashChange() {
  const hash = decodeURIComponent(window.location.hash.slice(1))
  if (hash === 'events') {
    currentPage.value = 'events'
    return
  }

  currentPage.value = 'requests'
  const found = requests.find((request) => request.name === hash)
  if (found) selectedRequest.value = found
}

async function connect() {
  error.value = null
  isConnecting.value = true
  try {
    if (isConnected.value) await obs.disconnect()
    await obs.connect(wsUrl.value, password.value, {
      eventSubscriptions: eventSubscriptionForPage(currentPage.value),
    })
    isConnected.value = true
  } catch (e) {
    error.value = e.message || 'Connection error'
  } finally {
    isConnecting.value = false
  }
}

async function disconnect() {
  error.value = null
  try {
    await obs.disconnect()
  } catch (e) {
    error.value = e.message
  }
}

// -------------------------------------------------------------------
// Sending requests
// -------------------------------------------------------------------
async function sendRequest() {
  if (!selectedRequest.value) return
  try {
    response.value = await obs.call(selectedRequest.value.name, buildData())
  } catch (e) {
    response.value = e
  }
}

// -------------------------------------------------------------------
// Param management
// -------------------------------------------------------------------
function addParam() {
  if (!selectedRequest.value) return
  paramState.value[selectedRequest.value.name].push({
    type: ParamType.STRING,
    name: '',
    value: '',
  })
}

function removeParam(param) {
  const arr = paramState.value[selectedRequest.value?.name]
  if (!arr) return
  const i = arr.indexOf(param)
  if (i !== -1) arr.splice(i, 1)
}

// -------------------------------------------------------------------
// Clipboard helpers
// -------------------------------------------------------------------
function copyObsRaw() {
  copy(
    JSON.stringify(
      buildRequestPayload('streamerbot_raw', selectedRequest.value?.name ?? '', buildData()),
    ),
  )
}

function copyCph() {
  copy(
    buildRequestPayload('streamerbot_raw_cph', selectedRequest.value?.name ?? '', buildData()),
  )
}

// -------------------------------------------------------------------
// Lifecycle
// -------------------------------------------------------------------
onMounted(() => {
  setupListeners()

  handleHashChange()
  window.addEventListener('hashchange', handleHashChange)

  connect()
})

onUnmounted(() => {
  window.removeEventListener('hashchange', handleHashChange)
  for (const [eventType, listener] of eventListeners) {
    obs.off(eventType, listener)
  }
  obs.disconnect()
})
</script>

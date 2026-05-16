<template>
  <!-- Toolbar -->
  <v-toolbar density="compact" color="#1d1d1d">
    <span class="d-flex flex-column mx-5">
      <span>OBS Raw Generator</span>
      <small class="text-grey">for Streamer.bot</small>
    </span>

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
          style="min-width: 75px"
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
    <v-row>
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
                    v-for="(param, index) in currentParams"
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
                        @click="removeParam(index)"
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
            <CodePreview :value="response" />
          </div>
        </div>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useLocalStorage, useClipboard } from '@vueuse/core'
import OBSWebSocket from 'obs-websocket-js'
import RequestMenu from './RequestMenu.vue'
import ObsParamInput from './ObsParamInput.vue'
import CodePreview from './CodePreview.vue'
import { requests, paramTypeItems, ParamType } from '../data/requests.js'

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

// -------------------------------------------------------------------
// Param value conversion
// -------------------------------------------------------------------
function convertValue(param) {
  if (param.type === ParamType.NUMBER) return Number(param.value)
  if (param.type === ParamType.BOOLEAN) return Boolean(param.value)
  if (param.type === ParamType.OBJECT) {
    if (param.params) {
      const obj = {}
      param.params.forEach((p) => (obj[p.name] = convertValue(p)))
      return obj
    }
    try {
      return JSON.parse(param.value)
    } catch {
      return {}
    }
  }
  return param.value
}

function buildData() {
  if (!selectedRequest.value || !currentParams.value) return {}
  const data = {}
  currentParams.value.forEach((p) => {
    data[p.name] = convertValue(p)
  })
  return data
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

// -------------------------------------------------------------------
// Connection
// -------------------------------------------------------------------
function setupListeners() {
  obs.on('ConnectionClosed', () => (isConnected.value = false))
  obs.on('ConnectionError', () => (isConnected.value = false))
}

async function connect() {
  error.value = null
  isConnecting.value = true
  try {
    if (isConnected.value) await obs.disconnect()
    await obs.connect(wsUrl.value, password.value)
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

function removeParam(index) {
  if (!selectedRequest.value) return
  paramState.value[selectedRequest.value.name].splice(index, 1)
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

  // Navigate to request from URL hash
  if (window.location.hash) {
    const name = window.location.hash.slice(1)
    const found = requests.find((r) => r.name === name)
    if (found) selectedRequest.value = found
  }

  connect()
})

onUnmounted(() => {
  obs.disconnect()
})
</script>

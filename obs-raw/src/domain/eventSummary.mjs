const omittedFallbackFields = /(?:Uuid|Settings|Transform|Data)$/

const labels = {
  canvasName: 'Canvas',
  filterIndex: 'Index',
  filterKind: 'Kind',
  filterName: 'Filter',
  inputKind: 'Kind',
  inputName: 'Input',
  mediaAction: 'Action',
  monitorType: 'Monitor',
  profileName: 'Profile',
  sceneCollectionName: 'Collection',
  sceneItemId: 'Item',
  sceneItemIndex: 'Index',
  sceneName: 'Scene',
  sourceName: 'Source',
  transitionDuration: 'Duration',
  transitionName: 'Transition',
  vendorName: 'Vendor',
}

function hasValue(value) {
  return value !== undefined && value !== null && value !== ''
}

function itemId(value) {
  return hasValue(value) ? `Item #${value}` : ''
}

function named(label, value) {
  return hasValue(value) ? `${label}: ${value}` : ''
}

function state(value, whenTrue, whenFalse) {
  if (typeof value !== 'boolean') return ''
  return value ? whenTrue : whenFalse
}

function join(...parts) {
  return parts.filter(Boolean).join(' · ')
}

function rename(label, oldName, newName) {
  if (!hasValue(oldName) && !hasValue(newName)) return ''
  if (!hasValue(oldName)) return named(label, newName)
  if (!hasValue(newName)) return named(label, oldName)
  return `${label}: ${oldName} → ${newName}`
}

function readableEnum(value) {
  if (!hasValue(value)) return ''
  return String(value)
    .replace(/^OBS_WEBSOCKET_OUTPUT_/, '')
    .replace(/^OBS_MEDIA_INPUT_ACTION_/, '')
    .replace(/^OBS_MONITORING_TYPE_/, '')
    .split('_')
    .filter(Boolean)
    .map((word) => word.charAt(0) + word.slice(1).toLowerCase())
    .join(' ')
}

function count(label, value) {
  return Array.isArray(value) ? `${label}: ${value.length}` : ''
}

function fallbackSummary(data) {
  return Object.entries(data)
    .filter(([key, value]) => !omittedFallbackFields.test(key) && hasValue(value))
    .map(([key, value]) => {
      if (Array.isArray(value)) return `${labels[key] ?? key}: ${value.length}`
      if (typeof value === 'object') return ''
      if (key === 'sceneItemId') return itemId(value)
      if (typeof value === 'boolean') return `${labels[key] ?? key}: ${value ? 'Yes' : 'No'}`
      return named(labels[key] ?? key, value)
    })
    .filter(Boolean)
    .slice(0, 3)
    .join(' · ')
}

export function summarizeEvent(type, data) {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return ''

  switch (type) {
    case 'CanvasNameChanged':
      return rename('Canvas', data.oldCanvasName, data.canvasName)
    case 'InputNameChanged':
      return rename('Input', data.oldInputName, data.inputName)
    case 'SceneNameChanged':
      return rename('Scene', data.oldSceneName, data.sceneName)
    case 'SourceFilterNameChanged':
      return join(
        named('Source', data.sourceName),
        rename('Filter', data.oldFilterName, data.filterName),
      )

    case 'SceneItemCreated':
    case 'SceneItemRemoved':
      return join(
        named('Scene', data.sceneName),
        named('Source', data.sourceName),
        itemId(data.sceneItemId),
      )
    case 'SceneItemEnableStateChanged':
      return join(
        named('Scene', data.sceneName),
        itemId(data.sceneItemId),
        state(data.sceneItemEnabled, 'Enabled', 'Disabled'),
      )
    case 'SceneItemLockStateChanged':
      return join(
        named('Scene', data.sceneName),
        itemId(data.sceneItemId),
        state(data.sceneItemLocked, 'Locked', 'Unlocked'),
      )
    case 'SceneItemSelected':
    case 'SceneItemTransformChanged':
      return join(named('Scene', data.sceneName), itemId(data.sceneItemId))
    case 'SceneItemListReindexed':
      return join(named('Scene', data.sceneName), count('Items', data.sceneItems))

    case 'CurrentProgramSceneChanged':
    case 'CurrentPreviewSceneChanged':
    case 'SceneCreated':
    case 'SceneRemoved':
      return named('Scene', data.sceneName)
    case 'SceneListChanged':
      return count('Scenes', data.scenes)

    case 'CurrentSceneTransitionChanged':
    case 'SceneTransitionStarted':
    case 'SceneTransitionEnded':
    case 'SceneTransitionVideoEnded':
      return named('Transition', data.transitionName)
    case 'CurrentSceneTransitionDurationChanged':
      return hasValue(data.transitionDuration) ? `Duration: ${data.transitionDuration} ms` : ''

    case 'SourceFilterCreated':
      return join(
        named('Source', data.sourceName),
        named('Filter', data.filterName),
        named('Kind', data.filterKind),
      )
    case 'SourceFilterRemoved':
    case 'SourceFilterSettingsChanged':
      return join(named('Source', data.sourceName), named('Filter', data.filterName))
    case 'SourceFilterEnableStateChanged':
      return join(
        named('Source', data.sourceName),
        named('Filter', data.filterName),
        state(data.filterEnabled, 'Enabled', 'Disabled'),
      )
    case 'SourceFilterListReindexed':
      return join(named('Source', data.sourceName), count('Filters', data.filters))

    case 'InputActiveStateChanged':
      return join(named('Input', data.inputName), state(data.videoActive, 'Active', 'Inactive'))
    case 'InputShowStateChanged':
      return join(named('Input', data.inputName), state(data.videoShowing, 'Showing', 'Hidden'))
    case 'InputMuteStateChanged':
      return join(named('Input', data.inputName), state(data.inputMuted, 'Muted', 'Unmuted'))
    case 'InputVolumeChanged':
      return join(
        named('Input', data.inputName),
        hasValue(data.inputVolumeDb) ? `${Number(data.inputVolumeDb).toFixed(1)} dB` : '',
      )
    case 'InputAudioBalanceChanged':
      return join(named('Input', data.inputName), named('Balance', data.inputAudioBalance))
    case 'InputAudioSyncOffsetChanged':
      return join(
        named('Input', data.inputName),
        hasValue(data.inputAudioSyncOffset) ? `Offset: ${data.inputAudioSyncOffset} ms` : '',
      )
    case 'InputAudioMonitorTypeChanged':
      return join(named('Input', data.inputName), named('Monitor', readableEnum(data.monitorType)))
    case 'MediaInputActionTriggered':
      return join(named('Input', data.inputName), named('Action', readableEnum(data.mediaAction)))

    case 'StreamStateChanged':
    case 'ReplayBufferStateChanged':
    case 'VirtualcamStateChanged':
      return join(
        named('State', readableEnum(data.outputState)),
        state(data.outputActive, 'Active', 'Inactive'),
      )
    case 'RecordStateChanged':
      return join(
        named('State', readableEnum(data.outputState)),
        state(data.outputActive, 'Active', 'Inactive'),
        named('Path', data.outputPath),
      )
    case 'RecordFileChanged':
      return named('Path', data.newOutputPath)
    case 'ReplayBufferSaved':
      return named('Path', data.savedReplayPath)
    case 'ScreenshotSaved':
      return named('Path', data.savedScreenshotPath)
    case 'StudioModeStateChanged':
      return state(data.studioModeEnabled, 'Studio mode enabled', 'Studio mode disabled')
    case 'VendorEvent':
      return join(named('Vendor', data.vendorName), named('Event', data.eventType))

    case 'SceneCollectionListChanged':
      return count('Collections', data.sceneCollections)
    case 'ProfileListChanged':
      return count('Profiles', data.profiles)
    case 'InputVolumeMeters':
      return count('Inputs', data.inputs)
    default:
      return fallbackSummary(data)
  }
}

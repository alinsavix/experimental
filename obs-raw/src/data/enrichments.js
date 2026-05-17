// Dynamic option helpers — receive (obs, ctx) where ctx = { ...request, params: currentParams }

async function getInputNames(obs) {
  return (await obs.call('GetInputList')).inputs.map((i) => i.inputName)
}

async function getSceneNames(obs) {
  return (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName)
}

async function getOutputNames(obs) {
  return (await obs.call('GetOutputList')).outputs.map((o) => o.outputName)
}

async function getTransitionNames(obs) {
  return (await obs.call('GetSceneTransitionList')).transitions.map((t) => t.transitionName)
}

async function getProfileNames(obs) {
  return (await obs.call('GetProfileList')).profiles
}

async function getInputKinds(obs) {
  return [...new Set((await obs.call('GetInputList')).inputs.map((i) => i.inputKind))]
}

async function getSupportedImageFormats(obs) {
  return (await obs.call('GetVersion')).supportedImageFormats
}

async function getFilterKinds(obs) {
  return (await obs.call('GetSourceFilterKindList')).sourceFilterKinds
}

async function getFilterNames(obs, ctx) {
  const sourceName = ctx?.params?.find((p) => p.name === 'sourceName')?.value
  if (!sourceName) return []
  return (await obs.call('GetSourceFilterList', { sourceName })).filters.map((f) => f.filterName)
}

async function getSceneItemSources(obs, ctx) {
  const sceneName = ctx?.params?.find((p) => p.name === 'sceneName')?.value
  if (!sceneName) return []
  return (await obs.call('GetSceneItemList', { sceneName })).sceneItems.map((i) => i.sourceName)
}

async function getSceneItemIds(obs, ctx) {
  const sceneName = ctx?.params?.find((p) => p.name === 'sceneName')?.value
  if (!sceneName) return []
  const sources = await getSceneItemSources(obs, ctx)
  const items = []
  for (const sourceName of sources) {
    const { sceneItemId } = await obs.call('GetSceneItemId', { sceneName, sourceName })
    items.push({ title: `${sceneItemId} (${sourceName})`, value: sceneItemId })
  }
  return items
}

// Per-request enrichments merged over protocol.json at build time.
// Keys are requestType names. Supported fields:
//   description  — overrides the protocol.json description (use sparingly)
//   params       — map of paramName → { getOptions?, getValue?, params?, ...overrides }
//
// getOptions(obs, ctx) → string[] | { title, value }[]   populate a select dropdown
// getValue(obs, ctx)   → any                             pre-fill the current value
// params               → ParamDefinition[]               sub-params for Object fields
//                        (protocol.json dotted fields are auto-nested; only needed
//                         when the protocol provides no dotted sub-field definitions)

export const enrichments = {
  // ---------------------------------------------------------------------------
  // General
  // ---------------------------------------------------------------------------

  GetVersion: {
    // Protocol description is about the plugin/RPC version rather than OBS itself
    description: 'Retrieve OBS version information',
  },

  GetStats: {
    // Protocol description is verbose; this is clearer in a menu context
    description: 'Retrieve various OBS statistics',
  },

  TriggerHotkeyByName: {
    params: {
      hotkeyName: {
        getOptions: async (obs) => (await obs.call('GetHotkeyList')).hotkeys,
      },
    },
  },

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------

  GetPersistentData: {
    params: {
      realm: {
        getOptions: async () => [
          'OBS_WEBSOCKET_DATA_REALM_GLOBAL',
          'OBS_WEBSOCKET_DATA_REALM_PROFILE',
        ],
      },
    },
  },

  SetPersistentData: {
    params: {
      realm: {
        getOptions: async () => [
          'OBS_WEBSOCKET_DATA_REALM_GLOBAL',
          'OBS_WEBSOCKET_DATA_REALM_PROFILE',
        ],
      },
    },
  },

  SetCurrentSceneCollection: {
    params: {
      sceneCollectionName: {
        getOptions: async (obs) => (await obs.call('GetSceneCollectionList')).sceneCollections,
      },
    },
  },

  SetCurrentProfile: {
    params: {
      profileName: { getOptions: getProfileNames },
    },
  },

  RemoveProfile: {
    params: {
      profileName: { getOptions: getProfileNames },
    },
  },

  // ---------------------------------------------------------------------------
  // Sources
  // ---------------------------------------------------------------------------

  GetSourceActive: {
    params: {
      sourceName: { getOptions: getInputNames },
    },
  },

  GetSourceScreenshot: {
    params: {
      sourceName: { getOptions: getInputNames },
      imageFormat: { getOptions: getSupportedImageFormats },
    },
  },

  SaveSourceScreenshot: {
    params: {
      sourceName: { getOptions: getInputNames },
      imageFormat: { getOptions: getSupportedImageFormats },
    },
  },

  // ---------------------------------------------------------------------------
  // Scenes
  // ---------------------------------------------------------------------------

  SetCurrentProgramScene: {
    params: {
      sceneName: { getOptions: getSceneNames },
    },
  },

  SetCurrentPreviewScene: {
    params: {
      sceneName: { getOptions: getSceneNames },
    },
  },

  RemoveScene: {
    params: {
      sceneName: { getOptions: getSceneNames },
    },
  },

  SetSceneName: {
    params: {
      sceneName: { getOptions: getSceneNames },
    },
  },

  GetSceneSceneTransitionOverride: {
    params: {
      sceneName: { getOptions: getSceneNames },
    },
  },

  SetSceneSceneTransitionOverride: {
    params: {
      sceneName: { getOptions: getSceneNames },
      transitionName: { getOptions: getTransitionNames },
    },
  },

  GetGroupSceneItemList: {
    // Protocol description is informal ("Basically GetSceneItemList, but for
    // groups") and includes developer discouragement advice that isn't
    // appropriate for a UI menu item
    description: 'Gets a list of all scene items in a group',
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetGroupList')).groups,
      },
    },
  },

  // ---------------------------------------------------------------------------
  // Inputs
  // ---------------------------------------------------------------------------

  GetInputList: {
    params: {
      inputKind: { getOptions: getInputKinds },
    },
  },

  CreateInput: {
    params: {
      sceneName: { getOptions: getSceneNames },
      inputKind: { getOptions: getInputKinds },
    },
  },

  RemoveInput: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  SetInputName: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  GetInputDefaultSettings: {
    params: {
      inputKind: { getOptions: getInputKinds },
    },
  },

  GetInputSettings: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  SetInputSettings: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  GetInputMute: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  SetInputMute: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  ToggleInputMute: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  GetInputVolume: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  SetInputVolume: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  GetInputAudioBalance: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  SetInputAudioBalance: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  GetInputAudioSyncOffset: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  SetInputAudioSyncOffset: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  GetInputAudioMonitorType: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  SetInputAudioMonitorType: {
    params: {
      inputName: { getOptions: getInputNames },
      monitorType: {
        getOptions: async () => [
          'OBS_MONITORING_TYPE_NONE',
          'OBS_MONITORING_TYPE_MONITOR_ONLY',
          'OBS_MONITORING_TYPE_MONITOR_AND_OUTPUT',
        ],
      },
    },
  },

  GetInputAudioTracks: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  SetInputAudioTracks: {
    params: {
      inputName: { getOptions: getInputNames },
      inputAudioTracks: {
        getValue: async (obs, ctx) => {
          const inputName = ctx?.params?.find((p) => p.name === 'inputName')?.value
          if (!inputName) return undefined
          return (await obs.call('GetInputAudioTracks', { inputName })).inputAudioTracks
        },
      },
    },
  },

  GetInputDeinterlaceMode: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  SetInputDeinterlaceMode: {
    params: {
      inputName: { getOptions: getInputNames },
      inputDeinterlaceMode: {
        getOptions: async () => [
          'OBS_DEINTERLACE_MODE_DISABLE',
          'OBS_DEINTERLACE_MODE_DISCARD',
          'OBS_DEINTERLACE_MODE_RETRO',
          'OBS_DEINTERLACE_MODE_BLEND',
          'OBS_DEINTERLACE_MODE_BLEND_2X',
          'OBS_DEINTERLACE_MODE_LINEAR',
          'OBS_DEINTERLACE_MODE_LINEAR_2X',
          'OBS_DEINTERLACE_MODE_YADIF',
          'OBS_DEINTERLACE_MODE_YADIF_2X',
        ],
      },
    },
  },

  GetInputDeinterlaceFieldOrder: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  SetInputDeinterlaceFieldOrder: {
    params: {
      inputName: { getOptions: getInputNames },
      inputDeinterlaceFieldOrder: {
        getOptions: async () => [
          'OBS_DEINTERLACE_FIELD_ORDER_TOP',
          'OBS_DEINTERLACE_FIELD_ORDER_BOTTOM',
        ],
      },
    },
  },

  GetInputPropertiesListPropertyItems: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  PressInputPropertiesButton: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  // ---------------------------------------------------------------------------
  // Outputs
  // ---------------------------------------------------------------------------

  GetOutputStatus: {
    params: {
      outputName: { getOptions: getOutputNames },
    },
  },

  ToggleOutput: {
    params: {
      outputName: { getOptions: getOutputNames },
    },
  },

  StartOutput: {
    params: {
      outputName: { getOptions: getOutputNames },
    },
  },

  StopOutput: {
    params: {
      outputName: { getOptions: getOutputNames },
    },
  },

  GetOutputSettings: {
    params: {
      outputName: { getOptions: getOutputNames },
    },
  },

  SetOutputSettings: {
    params: {
      outputName: { getOptions: getOutputNames },
    },
  },

  // ---------------------------------------------------------------------------
  // Transitions
  // ---------------------------------------------------------------------------

  SetCurrentSceneTransition: {
    params: {
      transitionName: { getOptions: getTransitionNames },
    },
  },

  // ---------------------------------------------------------------------------
  // Filters
  // ---------------------------------------------------------------------------

  GetSourceFilterList: {
    params: {
      sourceName: { getOptions: getInputNames },
    },
  },

  GetSourceFilterDefaultSettings: {
    params: {
      filterKind: { getOptions: getFilterKinds },
    },
  },

  CreateSourceFilter: {
    params: {
      sourceName: { getOptions: getInputNames },
      filterName: { getOptions: getFilterNames },
      filterKind: { getOptions: getFilterKinds },
    },
  },

  RemoveSourceFilter: {
    params: {
      sourceName: { getOptions: getInputNames },
      filterName: { getOptions: getFilterNames },
    },
  },

  SetSourceFilterName: {
    params: {
      sourceName: { getOptions: getInputNames },
      filterName: { getOptions: getFilterNames },
    },
  },

  GetSourceFilter: {
    params: {
      sourceName: { getOptions: getInputNames },
      filterName: { getOptions: getFilterNames },
    },
  },

  SetSourceFilterIndex: {
    params: {
      sourceName: { getOptions: getInputNames },
      filterName: { getOptions: getFilterNames },
    },
  },

  SetSourceFilterSettings: {
    params: {
      sourceName: { getOptions: getInputNames },
      filterName: { getOptions: getFilterNames },
    },
  },

  SetSourceFilterEnabled: {
    params: {
      sourceName: { getOptions: getInputNames },
      filterName: { getOptions: getFilterNames },
    },
  },

  // ---------------------------------------------------------------------------
  // Scene Items
  // ---------------------------------------------------------------------------

  GetSceneItemList: {
    params: {
      sceneName: { getOptions: getSceneNames },
    },
  },

  GetSceneItemId: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sourceName: { getOptions: getSceneItemSources },
    },
  },

  CreateSceneItem: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sourceName: { getOptions: getSceneItemSources },
    },
  },

  RemoveSceneItem: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  DuplicateSceneItem: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  GetSceneItemTransform: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  SetSceneItemTransform: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  GetSceneItemEnabled: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  SetSceneItemEnabled: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  GetSceneItemLocked: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  SetSceneItemLocked: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  GetSceneItemIndex: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  SetSceneItemIndex: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  GetSceneItemBlendMode: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  SetSceneItemBlendMode: {
    params: {
      sceneName: { getOptions: getSceneNames },
      sceneItemId: { getOptions: getSceneItemIds },
      sceneItemBlendMode: {
        getOptions: async () => [
          'OBS_BLEND_NORMAL',
          'OBS_BLEND_ADDITIVE',
          'OBS_BLEND_SUBTRACT',
          'OBS_BLEND_SCREEN',
          'OBS_BLEND_MULTIPLY',
          'OBS_BLEND_LIGHTEN',
          'OBS_BLEND_DARKEN',
        ],
      },
    },
  },

  GetSceneItemSource: {
    params: {
      sceneName: { getOptions: getSceneNames },
    },
  },

  // ---------------------------------------------------------------------------
  // Media Inputs
  // ---------------------------------------------------------------------------

  GetMediaInputStatus: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  SetMediaInputCursor: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  OffsetMediaInputCursor: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  TriggerMediaInputAction: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  // ---------------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------------

  OpenInputPropertiesDialog: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  OpenInputFiltersDialog: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  OpenInputInteractDialog: {
    params: {
      inputName: { getOptions: getInputNames },
    },
  },

  OpenVideoMixProjector: {
    params: {
      videoMixType: {
        getOptions: async () => [
          'OBS_WEBSOCKET_VIDEO_MIX_TYPE_PREVIEW',
          'OBS_WEBSOCKET_VIDEO_MIX_TYPE_PROGRAM',
          'OBS_WEBSOCKET_VIDEO_MIX_TYPE_MULTIVIEW',
        ],
      },
    },
  },

  OpenSourceProjector: {
    params: {
      sourceName: { getOptions: getInputNames },
    },
  },
}

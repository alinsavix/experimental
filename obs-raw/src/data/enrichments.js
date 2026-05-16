// Dynamic option helpers — receive (obs, ctx) where ctx = { ...request, params: currentParams }

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
            'OBS_WEBSOCKET_DATA_REALM_PROFILE'
        ],
      },
    },
  },

  SetPersistentData: {
    params: {
      realm: {
        getOptions: async () => [
            'OBS_WEBSOCKET_DATA_REALM_GLOBAL',
            'OBS_WEBSOCKET_DATA_REALM_PROFILE'
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
      profileName: {
        getOptions: async (obs) => (await obs.call('GetProfileList')).profiles,
      },
    },
  },

  RemoveProfile: {
    params: {
      profileName: {
        getOptions: async (obs) => (await obs.call('GetProfileList')).profiles,
      },
    },
  },

  // ---------------------------------------------------------------------------
  // Sources
  // ---------------------------------------------------------------------------

  GetSourceActive: {
    params: {
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  GetSourceScreenshot: {
    params: {
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      imageFormat: {
        getOptions: async (obs) => (await obs.call('GetVersion')).supportedImageFormats,
      },
    },
  },

  SaveSourceScreenshot: {
    params: {
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      imageFormat: {
        getOptions: async (obs) => (await obs.call('GetVersion')).supportedImageFormats,
      },
    },
  },

  // ---------------------------------------------------------------------------
  // Scenes
  // ---------------------------------------------------------------------------

  SetCurrentProgramScene: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    },
  },

  SetCurrentPreviewScene: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    },
  },

  RemoveScene: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    },
  },

  SetSceneName: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    },
  },

  GetSceneSceneTransitionOverride: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    },
  },

  SetSceneSceneTransitionOverride: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      transitionName: {
        getOptions: async (obs) =>
          (await obs.call('GetSceneTransitionList')).transitions.map((t) => t.transitionName),
      },
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
      inputKind: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputKind),
      },
    },
  },

  CreateInput: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      inputKind: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputKind),
      },
    },
  },

  RemoveInput: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  SetInputName: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  GetInputDefaultSettings: {
    params: {
      inputKind: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputKind),
      },
    },
  },

  GetInputSettings: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  SetInputSettings: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  GetInputMute: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  SetInputMute: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  ToggleInputMute: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  GetInputVolume: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  SetInputVolume: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  GetInputAudioBalance: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  SetInputAudioBalance: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  GetInputAudioSyncOffset: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  SetInputAudioSyncOffset: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  GetInputAudioMonitorType: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  SetInputAudioMonitorType: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
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
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  SetInputAudioTracks: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
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
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  SetInputDeinterlaceMode: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
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
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  SetInputDeinterlaceFieldOrder: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
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
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  PressInputPropertiesButton: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  // ---------------------------------------------------------------------------
  // Outputs
  // ---------------------------------------------------------------------------

  GetOutputStatus: {
    params: {
      outputName: {
        getOptions: async (obs) => (await obs.call('GetOutputList')).outputs.map((o) => o.outputName),
      },
    },
  },

  ToggleOutput: {
    params: {
      outputName: {
        getOptions: async (obs) => (await obs.call('GetOutputList')).outputs.map((o) => o.outputName),
      },
    },
  },

  StartOutput: {
    params: {
      outputName: {
        getOptions: async (obs) => (await obs.call('GetOutputList')).outputs.map((o) => o.outputName),
      },
    },
  },

  StopOutput: {
    params: {
      outputName: {
        getOptions: async (obs) => (await obs.call('GetOutputList')).outputs.map((o) => o.outputName),
      },
    },
  },

  GetOutputSettings: {
    params: {
      outputName: {
        getOptions: async (obs) => (await obs.call('GetOutputList')).outputs.map((o) => o.outputName),
      },
    },
  },

  SetOutputSettings: {
    params: {
      outputName: {
        getOptions: async (obs) => (await obs.call('GetOutputList')).outputs.map((o) => o.outputName),
      },
    },
  },

  // ---------------------------------------------------------------------------
  // Transitions
  // ---------------------------------------------------------------------------

  SetCurrentSceneTransition: {
    params: {
      transitionName: {
        getOptions: async (obs) =>
          (await obs.call('GetSceneTransitionList')).transitions.map((t) => t.transitionName),
      },
    },
  },

  // ---------------------------------------------------------------------------
  // Filters
  // ---------------------------------------------------------------------------

  GetSourceFilterList: {
    params: {
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  GetSourceFilterDefaultSettings: {
    params: {
      filterKind: {
        getOptions: async (obs) =>
          (await obs.call('GetSourceFilterList')).filters.map((f) => f.filterKind),
      },
    },
  },

  CreateSourceFilter: {
    params: {
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      filterName: { getOptions: getFilterNames },
      filterKind: {
        getOptions: async (obs) =>
          (await obs.call('GetSourceFilterList')).filters.map((f) => f.filterKind),
      },
    },
  },

  RemoveSourceFilter: {
    params: {
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      filterName: { getOptions: getFilterNames },
    },
  },

  SetSourceFilterName: {
    params: {
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      filterName: { getOptions: getFilterNames },
    },
  },

  GetSourceFilter: {
    params: {
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      filterName: { getOptions: getFilterNames },
    },
  },

  SetSourceFilterIndex: {
    params: {
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      filterName: { getOptions: getFilterNames },
    },
  },

  SetSourceFilterSettings: {
    params: {
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      filterName: { getOptions: getFilterNames },
    },
  },

  SetSourceFilterEnabled: {
    params: {
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      filterName: { getOptions: getFilterNames },
    },
  },

  // ---------------------------------------------------------------------------
  // Scene Items
  // ---------------------------------------------------------------------------

  GetSceneItemList: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    },
  },

  GetSceneItemId: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sourceName: { getOptions: getSceneItemSources },
    },
  },

  CreateSceneItem: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sourceName: { getOptions: getSceneItemSources },
    },
  },

  RemoveSceneItem: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  DuplicateSceneItem: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  GetSceneItemTransform: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  SetSceneItemTransform: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  GetSceneItemEnabled: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  SetSceneItemEnabled: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  GetSceneItemLocked: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  SetSceneItemLocked: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  GetSceneItemIndex: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  SetSceneItemIndex: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  GetSceneItemBlendMode: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      sceneItemId: { getOptions: getSceneItemIds },
    },
  },

  SetSceneItemBlendMode: {
    params: {
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
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
      sceneName: {
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    },
  },

  // ---------------------------------------------------------------------------
  // Media Inputs
  // ---------------------------------------------------------------------------

  GetMediaInputStatus: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  SetMediaInputCursor: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  OffsetMediaInputCursor: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  TriggerMediaInputAction: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  // ---------------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------------

  OpenInputPropertiesDialog: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  OpenInputFiltersDialog: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  OpenInputInteractDialog: {
    params: {
      inputName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },

  OpenVideoMixProjector: {
    params: {
      sceneItemBlendMode: {
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
      sourceName: {
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    },
  },
}

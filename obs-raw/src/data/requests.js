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

export const ParamType = {
  STRING: 'String',
  NUMBER: 'Number',
  BOOLEAN: 'Boolean',
  OBJECT: 'Object',
}

export const paramTypeItems = [
  { title: 'String', value: 'String' },
  { title: 'Number', value: 'Number' },
  { title: 'Boolean', value: 'Boolean' },
  { title: 'Object', value: 'Object' },
]

export const requests = [
  {
    name: 'GetVersion',
    description: 'Retrieve OBS version information',
    group: 'General',
  },
  {
    name: 'GetStats',
    description: 'Retrieve various OBS statistics',
    group: 'General',
  },
  {
    name: 'BroadcastCustomEvent',
    description: 'Broadcasts a CustomEvent to all WebSocket clients.',
    group: 'General',
    params: [
      {
        name: 'eventData',
        type: 'Object',
        description: 'Data payload to emit',
        required: true,
      },
    ],
  },
  {
    name: 'CallVendorRequest',
    description: 'Call a request registered to a vendor.',
    group: 'General',
    params: [
      {
        name: 'vendorName',
        type: 'String',
        description: 'Name of vendor',
        required: true,
      },
      {
        name: 'requestType',
        type: 'String',
        description: 'The request type to call',
        required: true,
      },
      {
        name: 'requestData',
        type: 'Object',
        description: 'Object containing request data',
        required: true,
      },
    ],
  },
  {
    name: 'GetHotkeyList',
    description: 'Retrieves a list of all available hotkey names.',
    group: 'General',
  },
  {
    name: 'TriggerHotkeyByName',
    description: 'Triggers a hotkey using its name.',
    group: 'General',
    params: [
      {
        name: 'hotkeyName',
        type: 'String',
        description: 'Name of hotkey to trigger',
        required: true,
        getOptions: async (obs) => (await obs.call('GetHotkeyList')).hotkeys,
      },
    ],
  },
  {
    name: 'TriggerHotkeyByKeySequence',
    description: 'Triggers a hotkey using a sequence of keys.',
    group: 'General',
    params: [
      {
        name: 'keyId',
        type: 'String',
        description: 'The OBS Key ID to use',
        required: true,
      },
      {
        name: 'keyModifiers',
        type: 'Object',
        description: 'Object containing key modifiers to apply.',
        required: false,
        params: [
          { name: 'shift', type: 'Boolean', description: 'Press Shift', required: false },
          { name: 'control', type: 'Boolean', description: 'Press CTRL', required: false },
          { name: 'alt', type: 'Boolean', description: 'Press ALT', required: false },
          { name: 'command', type: 'Boolean', description: 'Press CMD (Mac OS Only)', required: false },
        ],
      },
    ],
  },
  {
    name: 'Sleep',
    description:
      'Sleeps for a time duration or number of frames. Only available in request batches with types SERIAL_REALTIME or SERIAL_FRAME.',
    group: 'General',
    params: [
      {
        name: 'sleepMillis',
        type: 'Number',
        description: 'Number of milliseconds to sleep for (if `SERIAL_REALTIME` mode)',
        required: false,
      },
      {
        name: 'sleepFrames',
        type: 'Number',
        description: 'Number of frames to sleep for (if `SERIAL_FRAME` mode).',
        required: false,
      },
    ],
  },
  {
    name: 'GetPersistentData',
    description: 'Gets the value of a "slot" from the selected persistent data realm.',
    group: 'Configuration',
    params: [
      {
        name: 'realm',
        type: 'String',
        description: 'The data realm to select.',
        required: true,
        getOptions: async () => ['OBS_WEBSOCKET_DATA_REALM_GLOBAL', 'OBS_WEBSOCKET_DATA_REALM_PROFILE'],
      },
      {
        name: 'slotName',
        type: 'String',
        description: 'The name of the slot to retrieve data from.',
        required: true,
      },
    ],
  },
  {
    name: 'SetPersistentData',
    description: 'Sets the value of a "slot" from the selected persistent data realm.',
    group: 'Configuration',
    params: [
      {
        name: 'realm',
        type: 'String',
        description: 'The data realm to select.',
        required: true,
        getOptions: async () => ['OBS_WEBSOCKET_DATA_REALM_GLOBAL', 'OBS_WEBSOCKET_DATA_REALM_PROFILE'],
      },
      {
        name: 'slotName',
        type: 'String',
        description: 'The name of the slot to retrieve data from.',
        required: true,
      },
      {
        name: 'slotValue',
        type: 'String',
        description: 'The value to apply to the slot.',
        required: true,
      },
    ],
  },
  {
    name: 'GetSceneCollectionList',
    description: 'Gets an array of all scene collections.',
    group: 'Configuration',
  },
  {
    name: 'SetCurrentSceneCollection',
    description: 'Switches to a scene collection.',
    group: 'Configuration',
    params: [
      {
        name: 'sceneCollectionName',
        type: 'String',
        description: 'Name of the scene collection to switch to.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneCollectionList')).sceneCollections,
      },
    ],
  },
  {
    name: 'CreateSceneCollection',
    description:
      'Creates a new scene collection, switching to it in the process. Note: This will block until the collection has finished changing.',
    group: 'Configuration',
    params: [
      {
        name: 'sceneCollectionName',
        type: 'String',
        description: 'Name for the new scene collection.',
        required: true,
      },
    ],
  },
  {
    name: 'GetProfileList',
    description: 'Gets an array of all profiles.',
    group: 'Configuration',
  },
  {
    name: 'SetCurrentProfile',
    description: 'Switches to a profile.',
    group: 'Configuration',
    params: [
      {
        name: 'profileName',
        type: 'String',
        description: 'Name of the profile to switch to.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetProfileList')).profiles,
      },
    ],
  },
  {
    name: 'CreateProfile',
    description: 'Creates a new profile, switching to it in the process',
    group: 'Configuration',
    params: [
      {
        name: 'profileName',
        type: 'String',
        description: 'Name for the new profile.',
        required: true,
      },
    ],
  },
  {
    name: 'RemoveProfile',
    description:
      'Removes a profile. If the current profile is chosen, it will change to a different profile first.',
    group: 'Configuration',
    params: [
      {
        name: 'profileName',
        type: 'String',
        description: 'Name of the profile to remove.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetProfileList')).profiles,
      },
    ],
  },
  {
    name: 'GetProfileParameter',
    description: "Gets a parameter from the current profile's configureation",
    group: 'Configuration',
    params: [
      {
        name: 'parameterCategory',
        type: 'String',
        description: 'Category of the parameter to get.',
        required: true,
      },
      {
        name: 'parameterName',
        type: 'String',
        description: 'Name of the parameter to get.',
        required: true,
      },
    ],
  },
  {
    name: 'SetProfileParameter',
    description: "Sets the value of a parameter in the current profile's configuration.",
    group: 'Configuration',
    params: [
      {
        name: 'parameterCategory',
        type: 'String',
        description: 'Category of the parameter to set.',
        required: true,
      },
      {
        name: 'parameterName',
        type: 'String',
        description: 'Name of the parameter to set.',
        required: true,
      },
      {
        name: 'parameterValue',
        type: 'String',
        description: 'Value of the parameter to set. Use `null` to delete.',
        required: true,
      },
    ],
  },
  {
    name: 'GetVideoSettings',
    description: 'Gets the current video settings.',
    group: 'Configuration',
  },
  {
    name: 'SetVideoSettings',
    description:
      'Sets the current video settings. (Note: Fields must be in pairs. Ex. You cannot set baseWidth without also setting baseHeight.)',
    group: 'Configuration',
    params: [
      {
        name: 'fpsNumerator',
        type: 'Number',
        description: 'Numerator of the fractional FPS value.',
        required: false,
      },
      {
        name: 'fpsDenominator',
        type: 'Number',
        description: 'Denominator of the fractional FPS value.',
        required: false,
      },
      {
        name: 'baseWidth',
        type: 'Number',
        description: 'Width of the base (canvas) resolution in pixels.',
        required: false,
      },
      {
        name: 'baseHeight',
        type: 'Number',
        description: 'Height of the base (canvas) resolution in pixels.',
        required: false,
      },
      {
        name: 'outputWidth',
        type: 'Number',
        description: 'Width of the output resolution in pixels.',
        required: false,
      },
      {
        name: 'outputHeight',
        type: 'Number',
        description: 'Height of the output resolution in pixels.',
        required: false,
      },
    ],
  },
  {
    name: 'GetStreamServiceSettings',
    description: 'Gets the current stream service settings (stream destination).',
    group: 'Configuration',
  },
  {
    name: 'SetStreamServiceSettings',
    description: 'Sets the current stream service settings (stream destination).',
    group: 'Configuration',
    params: [
      {
        name: 'streamServiceType',
        type: 'String',
        description: 'Type of stream service to apply. Example: `rtmp_common` or `rtmp_custom`',
        required: true,
      },
      {
        name: 'streamServiceSettings',
        type: 'Object',
        description: 'Settings to apply to the service.',
        required: true,
      },
    ],
  },
  {
    name: 'GetSourceActive',
    description: 'Gets the active and show state of a source',
    group: 'Sources',
    params: [
      {
        name: 'sourceName',
        type: 'String',
        description: 'Source to get the state of',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'GetSourceScreenshot',
    description: 'Gets a Base64-encoded screenshot of a source',
    group: 'Sources',
    params: [
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source to take a screenshot of',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'imageFormat',
        type: 'String',
        description: 'Image compression format to use. Use `GetVersion` to get compatible image formats',
        required: true,
        getOptions: async (obs) => (await obs.call('GetVersion')).supportedImageFormats,
      },
      {
        name: 'imageWidth',
        type: 'Number',
        description: 'Width to scale the screenshot to',
        required: false,
      },
      {
        name: 'imageHeight',
        type: 'Number',
        description: 'Height to scale the screenshot to',
        required: false,
      },
      {
        name: 'imageCompressionQuality',
        type: 'Number',
        description: 'Compression quality to use. 0 for high compression, 100 for uncompressed.',
        required: false,
      },
    ],
  },
  {
    name: 'SaveSourceScreenshot',
    description: 'Saves a screenshot of a source to the filesystem',
    group: 'Sources',
    params: [
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source to take a screenshot of',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'imageFormat',
        type: 'String',
        description: 'Image compression format to use. Use `GetVersion` to get compatible image formats',
        required: true,
        getOptions: async (obs) => (await obs.call('GetVersion')).supportedImageFormats,
      },
      {
        name: 'imageFilePath',
        type: 'String',
        description: 'Path to save the screenshot file to. Example, `C:Users<user>Desktopscreenshot.png`',
        required: true,
      },
      {
        name: 'imageWidth',
        type: 'Number',
        description: 'Width to scale the screenshot to',
        required: false,
      },
      {
        name: 'imageHeight',
        type: 'Number',
        description: 'Height to scale the screenshot to',
        required: false,
      },
      {
        name: 'imageCompressionQuality',
        type: 'Number',
        description: 'Compression quality to use. 0 for high compression, 100 for uncompressed.',
        required: false,
      },
    ],
  },
  {
    name: 'GetSceneList',
    description: 'Gets an array of all scenes in OBS.',
    group: 'Scenes',
  },
  {
    name: 'GetGroupList',
    description: 'Gets an array of all groups in OBS',
    group: 'Scenes',
  },
  {
    name: 'GetCurrentProgramScene',
    description: 'Gets the current program scene',
    group: 'Scenes',
  },
  {
    name: 'SetCurrentProgramScene',
    description: 'Sets the current program scene',
    group: 'Scenes',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Scene to set as the current program scene.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    ],
  },
  {
    name: 'GetCurrentPreviewScene',
    description: 'Gets the current preview scene',
    group: 'Scenes',
  },
  {
    name: 'SetCurrentPreviewScene',
    description: 'Sets the current preview scene',
    group: 'Scenes',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Scene to set as the current preview scene.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    ],
  },
  {
    name: 'CreateScene',
    description: 'Creates a new scene in OBS',
    group: 'Scenes',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name for the new scene.',
        required: true,
      },
    ],
  },
  {
    name: 'RemoveScene',
    description: 'Removes a scene from OBS',
    group: 'Scenes',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene to remove.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    ],
  },
  {
    name: 'SetSceneName',
    description: 'Sets the name of a scene',
    group: 'Scenes',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene to be renamed',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'newSceneName',
        type: 'String',
        description: 'New name for the scene.',
        required: true,
      },
    ],
  },
  {
    name: 'GetSceneSceneTransitionOverride',
    description: 'Gets the scene transition overridden for a scene',
    group: 'Scenes',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    ],
  },
  {
    name: 'SetSceneSceneTransitionOverride',
    description: 'Sets the scene transition overridden for a scene',
    group: 'Scenes',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'transitionName',
        type: 'String',
        description: 'Name of the scene transition to use as override. Specify `null` to remove.',
        required: false,
        getOptions: async (obs) =>
          (await obs.call('GetSceneTransitionList')).transitions.map((t) => t.transitionName),
      },
      {
        name: 'transitionDuration',
        type: 'Number',
        description: 'Duration to use for any overridden transition. Specify `null` to remove.',
        required: false,
      },
    ],
  },
  {
    name: 'GetGroupSceneItemList',
    description: 'Gets a list of all scene item groups in a scene',
    group: 'Scenes',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the group to get the items of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetGroupList')).groups,
      },
    ],
  },
  {
    name: 'GetInputList',
    description: 'Gets an array of all inputs in OBS',
    group: 'Inputs',
    params: [
      {
        name: 'inputKind',
        type: 'String',
        description: 'Restrict the array to only inputs of the specified kind.',
        required: false,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputKind),
      },
    ],
  },
  {
    name: 'GetInputKindList',
    description: 'Gets an array of all available input kinds in OBS',
    group: 'Inputs',
    params: [
      {
        name: 'unversioned',
        type: 'Boolean',
        description:
          'true = Return all kinds as unversions, false = Return with version suffixes (if available).',
        required: false,
      },
    ],
  },
  {
    name: 'GetSpecialInputs',
    description: 'Gets the names of all special inputs',
    group: 'Inputs',
  },
  {
    name: 'CreateInput',
    description: 'Creates a new input, adding it as a scene item to the specified scene',
    group: 'Inputs',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene to add the input to as a scene item.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the new input to create.',
        required: true,
      },
      {
        name: 'inputKind',
        type: 'String',
        description: 'The kind of input to be created.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputKind),
      },
      {
        name: 'inputSettings',
        type: 'Object',
        description: 'Settings object to initialize the input with.',
        required: false,
      },
      {
        name: 'sceneItemEnabled',
        type: 'Boolean',
        description: 'Whether to set the created scene item to enabled or disabled.',
        required: false,
      },
    ],
  },
  {
    name: 'RemoveInput',
    description: 'Removes an existing input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to remove.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'SetInputName',
    description: 'Sets the name of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Current input name.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'newInputName',
        type: 'String',
        description: 'New name for the input.',
        required: true,
      },
    ],
  },
  {
    name: 'GetInputDefaultSettings',
    description: 'Gets the default settings for an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputKind',
        type: 'String',
        description: 'Input kind to get the default settings for.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputKind),
      },
    ],
  },
  {
    name: 'GetInputSettings',
    description: 'Gets the settings of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to get the settings of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'SetInputSettings',
    description: 'Sets the settings of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to set the settings of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'inputSettings',
        type: 'Object',
        description: 'Object of settings to apply.',
        required: true,
      },
      {
        name: 'overlay',
        type: 'Boolean',
        description:
          'true = apply the settings on top of existing ones, false = reset the input to its defaults, then apply settings.',
        required: false,
      },
    ],
  },
  {
    name: 'GetInputMute',
    description: 'Gets the audio mute state of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of input to get the mute state of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'SetInputMute',
    description: 'Sets the audio mute state of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to set the mute state of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'inputMuted',
        type: 'Boolean',
        description: 'Whether to mute the input or not.',
        required: true,
      },
    ],
  },
  {
    name: 'ToggleInputMute',
    description: 'Toggles the audio mute state of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to toggle the mute state of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'GetInputVolume',
    description: 'Gets the current volume setting of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to get the volume of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'SetInputVolume',
    description: 'Sets the volume setting of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to set the volume of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'inputVolumeMul',
        type: 'Number',
        description: 'Volume setting in mul.',
        required: false,
      },
      {
        name: 'inputVolumeDb',
        type: 'Number',
        description: 'Volume setting in dB.',
        required: false,
      },
    ],
  },
  {
    name: 'GetInputAudioBalance',
    description: 'Gets the audio balance of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to get the audio balance of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'SetInputAudioBalance',
    description: 'Sets the audio balance of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to set the audio balance of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'inputAudioBalance',
        type: 'Number',
        description: 'New audio balance value.',
        required: true,
      },
    ],
  },
  {
    name: 'GetInputAudioSyncOffset',
    description: 'Gets the audio sync offset of an input. Note: The audio synce offset can be negative too.',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to get the audio sync offset of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'SetInputAudioSyncOffset',
    description: 'Sets the audio sync offset of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to set the audio sync offset for.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'inputAudioSyncOffset',
        type: 'Number',
        description: 'New audio synce offset in milliseconds.',
        required: true,
      },
    ],
  },
  {
    name: 'GetInputAudioMonitorType',
    description: 'Gets the audio monitor type of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to set the audio monitor of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'SetInputAudioMonitorType',
    description: 'Sets the audio monitor type of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to set the audio monitor type for.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'monitorType',
        type: 'String',
        description: 'Audio monitor type.',
        required: true,
        getOptions: async () => [
          'OBS_MONITORING_TYPE_NONE',
          'OBS_MONITORING_TYPE_MONITOR_ONLY',
          'OBS_MONITORING_TYPE_MONITOR_AND_OUTPUT',
        ],
      },
    ],
  },
  {
    name: 'GetInputAudioTracks',
    description: 'Gets the enable state of all audio tracks of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'SetInputAudioTracks',
    description: 'Sets the enable state of audio tracks of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'inputAudioTracks',
        type: 'Object',
        description: 'Track settings to apply.',
        required: true,
        getValue: async (obs, ctx) => {
          const inputName = ctx?.params?.find((p) => p.name === 'inputName')?.value
          if (!inputName) return undefined
          return (await obs.call('GetInputAudioTracks', { inputName })).inputAudioTracks
        },
      },
    ],
  },
  {
    name: 'GetInputPropertiesListPropertyItems',
    description: 'Gets the items of a list property from an input properties',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'propertyName',
        type: 'String',
        description: 'Name of the list property to get the items of.',
        required: true,
      },
    ],
  },
  {
    name: 'PressInputPropertiesButton',
    description: 'Presses a button in the properties of an input',
    group: 'Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'propertyName',
        type: 'String',
        description: 'Name of the button property to press.',
        required: true,
      },
    ],
  },
  {
    name: 'GetTransitionKindList',
    description: 'Gets an array of all available transition kinds',
    group: 'Transitions',
  },
  {
    name: 'GetSceneTransitionList',
    description: 'Gets an array of all scene transitions in OBS',
    group: 'Transitions',
  },
  {
    name: 'GetCurrentSceneTransition',
    description: 'Gets information about the current scene transition',
    group: 'Transitions',
  },
  {
    name: 'SetCurrentSceneTransition',
    description: 'Sets the current scene transition',
    group: 'Transitions',
    params: [
      {
        name: 'transitionName',
        type: 'String',
        description: 'Name of the transition to make active.',
        required: true,
        getOptions: async (obs) =>
          (await obs.call('GetSceneTransitionList')).transitions.map((t) => t.transitionName),
      },
    ],
  },
  {
    name: 'SetCurrentSceneTransitionDuration',
    description: 'Sets the duration of the current scene transition',
    group: 'Transitions',
    params: [
      {
        name: 'transitionDuration',
        type: 'Number',
        description: 'Duration in milliseconds',
        required: true,
      },
    ],
  },
  {
    name: 'SetCurrentSceneTransitionSettings',
    description: 'Sets the settings of the current scene transition',
    group: 'Transitions',
    params: [
      {
        name: 'transitionSettings',
        type: 'Object',
        description: 'Settings object to apply to the transition. Can be {}',
        required: true,
      },
      {
        name: 'overlay',
        type: 'Boolean',
        description: 'Whether to overlay over the current settings or replace them.',
        required: false,
      },
    ],
  },
  {
    name: 'GetCurrentSceneTransitionCursor',
    description: 'Gets the cursor position of the current scene transition',
    group: 'Transitions',
  },
  {
    name: 'TriggerStudioModeTransition',
    description: 'Triggers the current scene transition',
    group: 'Transitions',
  },
  {
    name: 'SetTBarPosition',
    description: 'Sets the position of the TBar',
    group: 'Transitions',
    params: [
      {
        name: 'position',
        type: 'Number',
        description: 'New position.',
        required: true,
      },
      {
        name: 'release',
        type: 'Boolean',
        description:
          'Whether to release the TBar. Only set to false if you know that you will be sending another position update.',
        required: false,
      },
    ],
  },
  {
    name: 'GetSourceFilterList',
    description: 'Gets an array of all of a source filters',
    group: 'Filters',
    params: [
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'GetSourceFilterDefaultSettings',
    description: 'Gets the default settings for a filter',
    group: 'Filters',
    params: [
      {
        name: 'filterKind',
        type: 'String',
        description: 'Filter kind to get the default settings for.',
        required: true,
        getOptions: async (obs) =>
          (await obs.call('GetSourceFilterList')).filters.map((f) => f.filterKind),
      },
    ],
  },
  {
    name: 'CreateSourceFilter',
    description: 'Creates a new filter, adding it to the specified source',
    group: 'Filters',
    params: [
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source to add the filter to.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'filterName',
        type: 'String',
        description: 'Name of the new filter to be created.',
        required: true,
        getOptions: getFilterNames,
      },
      {
        name: 'filterKind',
        type: 'String',
        description: 'The kind of filter to be created.',
        required: true,
        getOptions: async (obs) =>
          (await obs.call('GetSourceFilterList')).filters.map((f) => f.filterKind),
      },
      {
        name: 'filterSettings',
        type: 'Object',
        description: 'Settings object to initialize the filter with.',
        required: false,
      },
    ],
  },
  {
    name: 'RemoveSourceFilter',
    description: 'Removes a filter from a source',
    group: 'Filters',
    params: [
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source the filter is on.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'filterName',
        type: 'String',
        description: 'Name of the filter to remove.',
        required: true,
        getOptions: getFilterNames,
      },
    ],
  },
  {
    name: 'SetSourceFilterName',
    description: 'Sets the name of a source filter',
    group: 'Filters',
    params: [
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source the filter is on.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'filterName',
        type: 'String',
        description: 'Current name of the filter.',
        required: true,
        getOptions: getFilterNames,
      },
      {
        name: 'newFilterName',
        type: 'String',
        description: 'New name of the filter.',
        required: true,
      },
    ],
  },
  {
    name: 'GetSourceFilter',
    description: 'Gets the info for a specific source filter',
    group: 'Filters',
    params: [
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'filterName',
        type: 'String',
        description: 'Name of the filter.',
        required: true,
        getOptions: getFilterNames,
      },
    ],
  },
  {
    name: 'SetSourceFilterIndex',
    description: 'Sets the index position of a filter on a source',
    group: 'Filters',
    params: [
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source the filter is on.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'filterName',
        type: 'String',
        description: 'Name of the filter.',
        required: true,
        getOptions: getFilterNames,
      },
      {
        name: 'filterIndex',
        type: 'Number',
        description: 'New index position of the filter.',
        required: true,
      },
    ],
  },
  {
    name: 'SetSourceFilterSettings',
    description: 'Sets the settings of a source filter',
    group: 'Filters',
    params: [
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source the filter is on.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'filterName',
        type: 'String',
        description: 'Name of the filter to set the settings of.',
        required: true,
        getOptions: getFilterNames,
      },
      {
        name: 'filterSettings',
        type: 'Object',
        description: 'Object of settings to apply.',
        required: true,
      },
      {
        name: 'overlay',
        type: 'Boolean',
        description:
          'true = apply the settings on top of existing ones, false = reset the input to its default, then apply settings.',
        required: false,
      },
    ],
  },
  {
    name: 'SetSourceFilterEnabled',
    description: 'Sets the enable state of a source filter',
    group: 'Filters',
    params: [
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source the filter is on.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'filterName',
        type: 'String',
        description: 'Name of the filter.',
        required: true,
        getOptions: getFilterNames,
      },
      {
        name: 'filterEnabled',
        type: 'Boolean',
        description: 'New enable state of the filter.',
        required: true,
      },
    ],
  },
  {
    name: 'GetSceneItemList',
    description: 'Gets a list of all scene items in a scene',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene to get the items of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
    ],
  },
  {
    name: 'GetSceneItemId',
    description: 'Searches a scene for a source, and returns its id',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene or group to search in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source to find.',
        required: true,
        getOptions: getSceneItemSources,
      },
      {
        name: 'searchOffset',
        type: 'Number',
        description:
          'Number of matches to skip during search. >=0 means first forward, -1 means last (top) item.',
        required: false,
      },
    ],
  },
  {
    name: 'CreateSceneItem',
    description: 'Creates a new scene item using a source',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene to create the new item in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sourceName',
        type: 'String',
        description: 'Name of the source to add to the scene.',
        required: true,
        getOptions: getSceneItemSources,
      },
      {
        name: 'sceneItemEnabled',
        type: 'Boolean',
        description: 'Enable state to apply to the scene item on creation.',
        required: false,
      },
    ],
  },
  {
    name: 'RemoveSceneItem',
    description: 'Removes a scene item from a scene',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
    ],
  },
  {
    name: 'DuplicateSceneItem',
    description: 'Duplicates a scene item, copying all transform and crop info',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
      {
        name: 'destinationSceneName',
        type: 'String',
        description: 'Name of the scene to create the dupicated item in. (`sceneName` is assumed)',
        required: false,
      },
    ],
  },
  {
    name: 'GetSceneItemTransform',
    description: 'Gets the transform and crop info of a scene item',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
    ],
  },
  {
    name: 'SetSceneItemTransform',
    description: 'Sets the transform and crop info of a scene item',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
      {
        name: 'sceneItemTransform',
        type: 'Object',
        description: 'Object containing scene item transform info to update.',
        required: true,
      },
    ],
  },
  {
    name: 'GetSceneItemEnabled',
    description: 'Gets the enable state of a scene item',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
    ],
  },
  {
    name: 'SetSceneItemEnabled',
    description: 'Sets the enable state of a scene item',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
      {
        name: 'sceneItemEnabled',
        type: 'Boolean',
        description: 'New enable state of the scene item.',
        required: true,
      },
    ],
  },
  {
    name: 'GetSceneItemLocked',
    description: 'Gets the lock state of a scene item',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
    ],
  },
  {
    name: 'SetSceneItemLocked',
    description: 'Sets the lock state of a scene item',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
      {
        name: 'sceneItemLocked',
        type: 'Boolean',
        description: 'New lock state of the scene item.',
        required: true,
      },
    ],
  },
  {
    name: 'GetSceneItemIndex',
    description: 'Gets the index position of a scene item in a scene',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
    ],
  },
  {
    name: 'SetSceneItemIndex',
    description: 'Sets the index position of a scene item in a scene',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
      {
        name: 'sceneItemIndex',
        type: 'Number',
        description: 'New index position of the scene item.',
        required: true,
      },
    ],
  },
  {
    name: 'GetSceneItemBlendMode',
    description: 'Gets the blend mode of a scene item',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
    ],
  },
  {
    name: 'SetSceneItemBlendMode',
    description: 'Sets the blend mode of a scene item',
    group: 'Scene Items',
    params: [
      {
        name: 'sceneName',
        type: 'String',
        description: 'Name of the scene the item is in.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetSceneList')).scenes.map((s) => s.sceneName),
      },
      {
        name: 'sceneItemId',
        type: 'Number',
        description: 'Numeric ID of the scene item.',
        required: true,
        getOptions: getSceneItemIds,
      },
      {
        name: 'sceneItemBlendMode',
        type: 'String',
        description: 'New blend mode.',
        required: true,
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
    ],
  },
  {
    name: 'GetVirtualCamStatus',
    description: 'Gets the status of the virtualcam output',
    group: 'Outputs',
  },
  {
    name: 'ToggleVirtualCam',
    description: 'Toggles the state of the virtualcam output',
    group: 'Outputs',
  },
  {
    name: 'StartVirtualCam',
    description: 'Starts the virtualcam output',
    group: 'Outputs',
  },
  {
    name: 'StopVirtualCam',
    description: 'Stops the virtualcam output',
    group: 'Outputs',
  },
  {
    name: 'GetReplayBufferStatus',
    description: 'Gets the status of the replay buffer output',
    group: 'Outputs',
  },
  {
    name: 'ToggleReplayBuffer',
    description: 'Toggles the state of the replay buffer output',
    group: 'Outputs',
  },
  {
    name: 'StartReplayBuffer',
    description: 'Starts the replay buffer output',
    group: 'Outputs',
  },
  {
    name: 'StopReplayBuffer',
    description: 'Stops the replay buffer output',
    group: 'Outputs',
  },
  {
    name: 'SaveReplayBuffer',
    description: 'Saves the contents of the replay buffer output',
    group: 'Outputs',
  },
  {
    name: 'GetLastReplayBufferReplay',
    description: 'Gets the filename of the last replay buffer save file',
    group: 'Outputs',
  },
  {
    name: 'GetStreamStatus',
    description: 'Gets the status of the stream output',
    group: 'Stream',
  },
  {
    name: 'ToggleStream',
    description: 'Toggles the status of the stream output',
    group: 'Stream',
  },
  {
    name: 'StartStream',
    description: 'Starts the stream output',
    group: 'Stream',
  },
  {
    name: 'StopStream',
    description: 'Stops the stream output',
    group: 'Stream',
  },
  {
    name: 'SendStreamCaption',
    description: 'Sends CEA-608 caption text over the stream output',
    group: 'Stream',
    params: [
      {
        name: 'captionText',
        type: 'String',
        description: 'Caption text.',
        required: true,
      },
    ],
  },
  {
    name: 'GetRecordStatus',
    description: 'Gets the status of the record output',
    group: 'Record',
  },
  {
    name: 'ToggleRecord',
    description: 'Toggles the status of the record output',
    group: 'Record',
  },
  {
    name: 'StartRecord',
    description: 'Starts the record output',
    group: 'Record',
  },
  {
    name: 'StopRecord',
    description: 'Stops the record output',
    group: 'Record',
  },
  {
    name: 'ToggleRecordPause',
    description: 'Toggles pause on the record output',
    group: 'Record',
  },
  {
    name: 'PauseRecord',
    description: 'Pauses the record output',
    group: 'Record',
  },
  {
    name: 'ResumeRecord',
    description: 'Resumes the record output',
    group: 'Record',
  },
  {
    name: 'GetMediaInputStatus',
    description: 'Gets the status of a media input',
    group: 'Media Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the media input.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'SetMediaInputCursor',
    description: 'Sets the cursor position of a media input',
    group: 'Media Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the media input.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'mediaCursor',
        type: 'Number',
        description: 'New cursor position to set.',
        required: true,
      },
    ],
  },
  {
    name: 'OffsetMediaInputCursor',
    description: 'Offsets the current cursor position of a media input by the specified value',
    group: 'Media Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the media input.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'mediaCursorOffset',
        type: 'Number',
        description: 'Value to offset the current cursor position by.',
        required: true,
      },
    ],
  },
  {
    name: 'TriggerMediaInputAction',
    description: 'Triggers an action on a media input',
    group: 'Media Inputs',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the media input.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
      {
        name: 'mediaAction',
        type: 'String',
        description: 'Identifier of the `ObsMediaInputAction` enum.',
        required: true,
      },
    ],
  },
  {
    name: 'GetStudioModeEnabled',
    description: 'Gets whether studio is enabled',
    group: 'UI',
  },
  {
    name: 'SetStudioModeEnabled',
    description: 'Enables or disables studio mode',
    group: 'UI',
    params: [
      {
        name: 'studioModeEnabled',
        type: 'Boolean',
        description: 'true = enabled, false = disabled.',
        required: true,
      },
    ],
  },
  {
    name: 'OpenInputPropertiesDialog',
    description: 'Opens the properties dialog of an input',
    group: 'UI',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to open the dialog of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'OpenInputFiltersDialog',
    description: 'Opens the filters dialog of an input',
    group: 'UI',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to open the dialog of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'OpenInputInteractDialog',
    description: 'Opens the interact dialog of an input',
    group: 'UI',
    params: [
      {
        name: 'inputName',
        type: 'String',
        description: 'Name of the input to open the dialog of.',
        required: true,
        getOptions: async (obs) => (await obs.call('GetInputList')).inputs.map((i) => i.inputName),
      },
    ],
  },
  {
    name: 'GetMonitorList',
    description: 'Gets a list of connected monitors and information about them',
    group: 'UI',
  },
]

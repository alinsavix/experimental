import proto from './protocol.json'
import { enrichments } from './enrichments.js'

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

function categoryToGroup(cat) {
  if (cat === 'config') return 'Configuration'
  if (cat === 'ui') return 'UI'
  return cat.replace(/\b\w/g, (c) => c.toUpperCase())
}

// Order of sections in the left-hand request list. Edit this to reorder them.
// Names must match the group titles (see categoryToGroup above). Any group not
// listed here is placed at the bottom, keeping its original protocol order.
export const groupOrder = [
  'General',
  'Configuration',
  'Sources',
  'Scenes',
  'Scene Items',
  'Inputs',
  'Transitions',
  'Filters',
  'Outputs',
  'Stream',
  'Record',
  'Media Inputs',
  'UI',
]

function groupRank(group) {
  const i = groupOrder.indexOf(group)
  return i === -1 ? Infinity : i
}

// Params that only apply to newer OBS versions. Maps a param name to a request
// that must be present in OBS's availableRequests for the param to be relevant.
// Such params are hidden when connected to an OBS that lacks that request.
// e.g. canvasUuid only matters where multi-canvas support exists (GetCanvasList).
export const paramRequiresRequest = {
  canvasUuid: 'GetCanvasList',
}

function mapType(t) {
  return t === 'Any' ? 'Object' : t
}

function buildParams(requestFields, paramEnrichments = {}) {
  // Separate top-level fields from dotted sub-fields (e.g. keyModifiers.shift)
  const topLevel = requestFields.filter((f) => !f.valueName.includes('.'))
  const subFields = requestFields.filter((f) => f.valueName.includes('.'))

  return topLevel.map((f) => {
    const enrichment = paramEnrichments[f.valueName] ?? {}
    const param = {
      name: f.valueName,
      type: mapType(f.valueType),
      description: f.valueDescription,
      required: !f.valueOptional,
      ...enrichment,
    }

    // Tag version-gated params (e.g. canvasUuid) so the UI can hide them when
    // connected to an OBS that doesn't support the corresponding feature.
    if (paramRequiresRequest[f.valueName] && !param.requiresRequest) {
      param.requiresRequest = paramRequiresRequest[f.valueName]
    }

    // Auto-nest dotted sub-fields under their parent Object field, unless the
    // enrichment already provides an explicit params array
    if (f.valueType === 'Object' && !enrichment.params) {
      const prefix = f.valueName + '.'
      const nested = subFields
        .filter((sf) => sf.valueName.startsWith(prefix))
        .map((sf) => ({
          name: sf.valueName.slice(prefix.length),
          type: mapType(sf.valueType),
          description: sf.valueDescription,
          required: !sf.valueOptional,
        }))
      if (nested.length) param.params = nested
    }

    return param
  })
}

export const requests = proto.requests
  .map((r) => {
    const enrichment = enrichments[r.requestType] ?? {}
    return {
      name: r.requestType,
      description: enrichment.description ?? r.description,
      group: categoryToGroup(r.category),
      ...(r.requestFields.length
        ? { params: buildParams(r.requestFields, enrichment.params) }
        : {}),
    }
  })
  // Stable sort by groupOrder; within a group the original order is preserved,
  // which keeps each group's requests contiguous for the list subheaders.
  .sort((a, b) => groupRank(a.group) - groupRank(b.group))

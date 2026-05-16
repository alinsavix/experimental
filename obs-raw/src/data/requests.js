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

export const requests = proto.requests.map((r) => {
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

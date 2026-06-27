const ABSENT = Symbol('absent request field')

function convertValue(param) {
  if (param.type === 'Object' && param.params) {
    const value = buildRequestData(param.params)
    return Object.keys(value).length ? value : ABSENT
  }

  if (param.value == null) return ABSENT
  if (param.type === 'Number' && String(param.value).trim() === '') return ABSENT
  if (param.allowsNull && param.value === 'null') return null

  if (param.type === 'Number') return Number(param.value)
  if (param.type === 'Boolean') return Boolean(param.value)
  if (param.type === 'Object') {
    try {
      return JSON.parse(param.value)
    } catch {
      return {}
    }
  }
  return param.value
}

export function buildRequestData(params) {
  const data = {}

  for (const param of params) {
    const value = convertValue(param)
    if (value !== ABSENT) data[param.name] = value
  }

  return data
}

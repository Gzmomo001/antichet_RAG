import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

export const getHealth = () =>
  api.get('/health').then((r) => r.data)

export const analyzeText = (text, source = 'user_submission') =>
  api.post('/analyze', { text, source }).then((r) => r.data)

export const addCase = (data) =>
  api.post('/cases', data).then((r) => r.data)

export const addTip = (data) =>
  api.post('/tips', data).then((r) => r.data)

export const searchCases = (query, limit = 5) =>
  api.get('/cases/search', { params: { query, limit } }).then((r) => r.data)

export const hybridSearch = (query, limit = 10) =>
  api.get('/search', { params: { query, limit } }).then((r) => r.data)

export default api

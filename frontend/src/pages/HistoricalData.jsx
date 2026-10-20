import { useEffect, useState } from "react"
import { Link } from "react-router-dom"
import GroundwaterChart from "../components/GroundwaterChart"
import { checkHealth, listStations, getHistoricalReadings, formatLevel } from "../api"

export default function HistoricalData() {
  const [health, setHealth] = useState(null)
  const [stations, setStations] = useState([])
  const [readings, setReadings] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedStation, setSelectedStation] = useState("Pune (Station 1234)")

  // Load health and data on mount
  useEffect(() => {
    async function loadData() {
      try {
        const h = await checkHealth()
        setHealth(h)
        const st = await listStations()
        setStations(st)
        const rd = await getHistoricalReadings()
        setReadings(rd)
      } catch (e) {
        console.error("Failed to load historical data:", e)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  if (loading) {
    return (
      <div className="page-historical min-h-screen text-slate-800 antialiased pb-12">
        <div className="flex items-center justify-center h-full">
          <span className="text-slate-500">Loading...</span>
        </div>
      </div>
    )
  }

  // Handle station change
  const handleStationChange = (e) => {
    setSelectedStation(e.target.value)
  }

  return (
    <div className="page-historical min-h-screen text-slate-800 antialiased pb-12">
      {/* BEGIN: MainHeader */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs" data-purpose="top-navigation">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Brand Logo */}
          <div className="flex items-center gap-2.5">
            <div className="text-blue-600 flex items-center justify-center">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 32 32">
                <path d="M16 2.5C16 2.5 6 14.5 6 21C6 26.5228 10.4772 31 16 31C21.5228 31 26 26.5228 26 21C26 14.5 16 2.5 16 2.5Z" stroke="#2563EB" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path>
                <path d="M16 8C16 8 10 16 10 20.5C10 23.8 12.7 26.5 16 26.5C19.3 26.5 22 23.8 22 20.5C22 16 16 8 16 8Z" fill="#2563EB" fillOpacity="0.3"></path>
              </svg>
            </div>
            <span className="text-2xl font-bold tracking-tight text-blue-600">HydraSense</span>
          </div>
          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-8 text-sm font-medium">
            <Link className="text-slate-600 hover:text-blue-600 transition-colors py-5" to="/">Home</Link>
            <Link className="text-slate-600 hover:text-blue-600 transition-colors py-5" to="/dashboard">Dashboard</Link>
            <Link className="text-blue-600 border-b-2 border-blue-600 py-5 font-semibold" to="/historical-data">Historical Data</Link>
            <Link className="text-slate-600 hover:text-blue-600 transition-colors py-5" to="/predictions">Predictions</Link>
            <Link className="text-slate-600 hover:text-blue-600 transition-colors py-5" to="/about">About</Link>
          </nav>
          {/* Right Location Dropdown Pill */}
          <div className="flex items-center">
            <button className="flex items-center gap-2 bg-white hover:bg-slate-50 border border-slate-300 rounded-lg px-3.5 py-1.5 text-sm font-medium text-slate-700 shadow-xs transition" type="button">
              <svg className="w-4 h-4 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
                <path d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
              </svg>
              <span>Pune, Maharashtra</span>
              <svg className="w-3.5 h-3.5 text-slate-500 ml-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path>
              </svg>
            </button>
            <select
              className="ml-4 appearance-none bg-white border border-slate-300 rounded-lg py-2 pl-3 pr-8 text-sm font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              value={selectedStation}
              onChange={handleStationChange}
            >
              <option>Pune (Station 1234)</option>
              <option>Haveli (Station 5678)</option>
              <option>Kothrud (Station 9012)</option>
            </select>
          </div>
        </div>
      </header>
      {/* END: MainHeader }}.
      {/* BEGIN: MainContent */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 pb-10">
        {/* Page Title & Description */}
        <div className="mb-6">
          <div className="flex items-center gap-3">
            {/* Blue bar chart icon */}
            <div className="text-blue-600">
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                <rect height="9" rx="1" width="4" x="3" y="12"></rect>
                <rect height="14" rx="1" width="4" x="10" y="7"></rect>
                <rect height="18" rx="1" width="4" x="17" y="3"></rect>
              </svg>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">Historical Data</h1>
          </div>
          <p className="mt-1 text-sm text-slate-500 pl-11">View past groundwater level records for the selected station.</p>
        </div>
        {/* BEGIN: FiltersBar */}
        <section className="bg-white rounded-xl shadow-xs border border-slate-200/90 p-4 mb-6" data-purpose="filters-section">
          <div className="flex flex-col lg:flex-row items-stretch lg:items-end justify-between gap-4">
            {/* Selection Filters */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 flex-1">
              {/* State Selector */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">State</label>
                <div className="relative">
                  <select className="w-full appearance-none bg-white border border-slate-300 rounded-lg py-2 pl-3 pr-8 text-sm font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                    <option>Maharashtra</option>
                    <option>Karnataka</option>
                    <option>Gujarat</option>
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2.5 text-slate-500">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path></svg>
                  </div>
                </div>
              </div>
              {/* District Selector */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">District</label>
                <div className="relative">
                  <select className="w-full appearance-none bg-white border border-slate-300 rounded-lg py-2 pl-3 pr-8 text-sm font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                    <option>Pune</option>
                    <option>Mumbai</option>
                    <option>Nagpur</option>
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2.5 text-slate-500">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path></svg>
                  </div>
                </div>
              </div>
              {/* Station Selector */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Station</label>
                <div className="relative">
                  <select className="w-full appearance-none bg-white border border-slate-300 rounded-lg py-2 pl-3 pr-8 text-sm font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                    <option>Pune (Station 1234)</option>
                    <option>Haveli (Station 5678)</option>
                    <option>Kothrud (Station 9012)</option>
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2.5 text-slate-500">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path></svg>
                  </div>
                </div>
              </div>
            </div>
            {/* Search Button and Date Range Selector */}
            <div className="flex items-center gap-3">
              {/* Search Button */}
              <button className="inline-flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-medium text-sm px-5 py-2 rounded-lg shadow-xs transition duration-150 ease-in-out cursor-pointer" type="button">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path>
                </svg>
                <span>Search</span>
              </button>
              {/* Date Range Dropdown Pill */}
              <div className="relative">
                <button className="flex items-center gap-2.5 bg-white border border-slate-300 rounded-lg px-3.5 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 transition shadow-xs" type="button">
                  <svg className="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
                  </svg>
                  <span>Last 1 Year</span>
                  <svg className="w-3.5 h-3.5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path>
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </section>
        {/* END: FiltersBar }}.
        {/* BEGIN: GroundwaterChartCard */}
        <section className="bg-white rounded-xl shadow-xs border border-slate-200/90 p-5 mb-6" data-purpose="chart-container">
          {/* Card Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 gap-3">
            {/* Title with waves icon */}
            <div className="flex items-center gap-2.5">
              <div className="text-blue-500">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M3 10c2.5-3 4.5-3 7 0s4.5 3 7 0 4.5-3 7 0M3 15c2.5-3 4.5-3 7 0s4.5 3 7 0 4.5-3 7 0" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path>
                </svg>
              </div>
              <h2 className="text-base sm:text-lg font-bold text-slate-900">Historical Groundwater Levels</h2>
            </div>
            {/* Legend */}
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 bg-slate-50/70 border border-slate-100 rounded-md px-2.5 py-1">
              <div className="flex items-center">
                <span className="w-4 h-0.5 bg-blue-600 inline-block"></span>
                <span className="w-2.5 h-2.5 bg-blue-600 rounded-full -ml-1.5 inline-block"></span>
                <span className="w-4 h-0.5 bg-blue-600 -ml-0.5 inline-block"></span>
              </div>
              <span>Water Level (m BGL)</span>
            </div>
          </div>
          {/* Chart Display Area */}
          <div className="relative w-full h-[320px] pt-2">
            <GroundwaterChart />
          </div>
        </section>
        {/* END: GroundwaterChartCard }}.
        {/* BEGIN: DataTableCard */}
        <section className="bg-white rounded-xl shadow-xs border border-slate-200/90 overflow-hidden" data-purpose="table-container">
          {/* Table Card Header */}
          <div className="px-5 py-4 flex items-center gap-2.5">
            <div className="text-blue-600">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s-8-1.79-8-4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
              </svg>
            </div>
            <h2 className="text-base sm:text-lg font-bold text-slate-900">Data Records</h2>
          </div>
          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-blue-50/70 border-y border-blue-100/80">
                  <th className="py-3 px-6 text-xs font-bold text-slate-800 tracking-normal w-1/3" scope="col">Date</th>
                  <th className="py-3 px-6 text-xs font-bold text-slate-800 tracking-normal w-1/3" scope="col">Water Level (m BGL)</th>
                  <th className="py-3 px-6 text-xs font-bold text-slate-800 tracking-normal w-1/3" scope="col">Rainfall (mm)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm font-medium text-slate-700">
                {readings.map((reading, i) => (
                  <tr key={i} className="hover:bg-slate-50/60 transition-colors">
                    <td className="py-3 px-6">{reading.date}</td>
                    <td className="py-3 px-6">{reading.water_level_m !== null ? formatLevel(reading.water_level_m, true) : "N/A"}</td>
                    <td className="py-3 px-6">{reading.rainfall_mm !== null ? reading.rainfall_mm + " mm" : "N/A"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
        {/* END: DataTableCard */}
      </main>
      {/* END: MainContent */}
    </div>
  )
}
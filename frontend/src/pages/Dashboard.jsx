import { useEffect, useState } from "react"
import { Link } from "react-router-dom"
import { checkHealth, listStations, getHistoricalReadings, formatLevel } from "../api"

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [stations, setStations] = useState([])
  const [readings, setReadings] = useState([])
  const [loading, setLoading] = useState(true)

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
        console.error("Failed to load dashboard data:", e)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  if (loading) {
    return (
      <div className="page-dashboard min-h-screen text-slate-800 antialiased p-3 sm:p-6 lg:p-8">
        <div className="flex items-center justify-center h-full">
          <span className="text-slate-500">Loading...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="page-dashboard min-h-screen text-slate-800 antialiased p-3 sm:p-6 lg:p-8">
      {/* Master App Container */}
      <div className="max-w-[1440px] mx-auto space-y-6">
        {/* BEGIN: TopNavigationBar */}
        <header className="bg-white rounded-2xl px-6 py-3.5 flex flex-wrap items-center justify-between shadow-sm border border-slate-100">
          {/* Brand Logo */}
          <div className="flex items-center gap-2.5">
            <div className="text-blue-600">
              <svg className="w-8 h-8 drop-shadow-sm" fill="none" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
                <path d="M16 2.5C16 2.5 6 14.5 6 21C6 26.5228 10.4772 31 16 31C21.5228 31 26 26.5228 26 21C26 14.5 16 2.5 16 2.5Z" fill="#1d72fe"></path>
                <path d="M16 7C16 7 10 16 10 21C10 24.3137 12.6863 27 16 27C19.3137 27 22 24.3137 22 21C22 16 16 7 16 7Z" fill="white" fillOpacity="0.25"></path>
              </svg>
            </div>
            <span className="text-2xl font-bold tracking-tight text-slate-900">
              Hydra<span className="text-blue-600">Sense</span>
            </span>
          </div>
          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-9 text-[15px] font-medium text-slate-600">
            <Link className="hover:text-blue-600 transition-colors" to="/">Home</Link>
            <Link className="text-blue-600 font-semibold relative py-1" to="/dashboard">
              Dashboard
              <span className="absolute bottom-[-14px] left-0 right-0 h-[2.5px] bg-blue-600 rounded-full"></span>
            </Link>
            <Link className="hover:text-blue-600 transition-colors" to="/historical-data">Historical Data</Link>
            <Link className="hover:text-blue-600 transition-colors" to="/predictions">Predictions</Link>
            <Link className="hover:text-blue-600 transition-colors" to="/about">About</Link>
          </nav>
          {/* Global Location Context */}
          <div className="flex items-center">
            <button className="flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-slate-200 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors" type="button">
              <svg className="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
                <path d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
              </svg>
              <span>Pune, Maharashtra</span>
              <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
              </svg>
            </button>
          </div>
        </header>
        {/* END: TopNavigationBar }}

        {/* BEGIN: DashboardHeaderAndFilters */}
        <section className="space-y-4" data-purpose="filters-section">
          {/* Title & Subtitle */}
          <div className="px-1">
            <h1 className="text-3xl font-extrabold tracking-tight text-slate-900">Dashboard</h1>
            <p className="text-base text-slate-500 mt-1 font-normal">Overview of groundwater level and key insights.</p>
          </div>
          {/* Station Selector Bar */}
          <div className="bg-white rounded-2xl p-4 shadow-sm border border-slate-100 grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
            {/* State Selector */}
            <div className="md:col-span-3">
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 ml-1">State</label>
              <div className="relative">
                <select className="w-full appearance-none bg-slate-50/70 border border-slate-200 rounded-xl px-4 py-2.5 text-sm font-medium text-slate-800 pr-10 hover:border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                  <option>Maharashtra</option>
                  <option>Gujarat</option>
                  <option>Karnataka</option>
                  <option>Rajasthan</option>
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3.5 text-slate-500">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path></svg>
                </div>
              </div>
            </div>
            {/* District Selector */}
            <div className="md:col-span-3">
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 ml-1">District</label>
              <div className="relative">
                <select className="w-full appearance-none bg-slate-50/70 border border-slate-200 rounded-xl px-4 py-2.5 text-sm font-medium text-slate-800 pr-10 hover:border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                  <option>Pune</option>
                  <option>Satara</option>
                  <option>Nashik</option>
                  <option>Nagpur</option>
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3.5 text-slate-500">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path></svg>
                </div>
              </div>
            </div>
            {/* Station Selector */}
            <div className="md:col-span-4">
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 ml-1">Station</label>
              <div className="relative">
                <select className="w-full appearance-none bg-slate-50/70 border border-slate-200 rounded-xl px-4 py-2.5 text-sm font-medium text-slate-800 pr-10 hover:border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                  <option>Pune (Station 2301)</option>
                  <option>Haveli (Station 1042)</option>
                  <option>Shivajinagar (Station 3105)</option>
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3.5 text-slate-500">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path></svg>
                </div>
              </div>
            </div>
            {/* Search Button */}
            <div className="md:col-span-2">
              <button className="w-full bg-[#0066ff] hover:bg-[#0052cc] text-white font-medium py-2.5 px-4 rounded-xl flex items-center justify-center gap-2 shadow-sm transition-all focus:ring-2 focus:ring-offset-2 focus:ring-blue-600" type="button">
                <svg className="w-4 h-4 stroke-[2.5]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" strokeLinecap="round" strokeLinejoin="round"></path>
                </svg>
                <span className="text-sm font-semibold">Search</span>
              </button>
            </div>
          </div>
        </section>
        {/* END: DashboardHeaderAndFilters }}

        {/* BEGIN: KeyMetricsCards */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4" data-purpose="metrics-summary">
          {/* Card 1: Current Level */}
          <article className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100 flex items-start space-x-4">
            <div className="w-14 h-14 rounded-2xl bg-blue-50 flex items-center justify-center shrink-0 text-blue-600">
              <svg className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M12 21.5c4.142 0 7.5-3.358 7.5-7.5 0-5-7.5-12-7.5-12S4.5 9 4.5 14c0 4.142 3.358 7.5 7.5 7.5z" strokeLinecap="round" strokeLinejoin="round"></path>
              </svg>
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Current Level</p>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-2xl font-bold tracking-tight text-slate-900">14.8 m</span>
                <span className="inline-flex items-center text-xs font-semibold text-emerald-600">
                  <svg className="w-3.5 h-3.5 mr-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 14l-7 7m0 0l-7-7m7 7V3" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path></svg>
                  0.7 m
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium mt-1">Below Ground Level (BGL)</p>
              <p className="text-[11px] text-slate-400 mt-0.5">Last updated: 10 Mar 2025</p>
            </div>
          </article>
          {/* Card 2: Predicted Level */}
          <article className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100 flex items-start space-x-4">
            <div className="w-14 h-14 rounded-2xl bg-emerald-50 flex items-center justify-center shrink-0 text-emerald-600">
              <svg className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24">
                <path d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" strokeLinecap="round" strokeLinejoin="round"></path>
              </svg>
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Predicted Level <span className="text-slate-400 capitalize font-normal">(Next 3 Mos)</span></p>
              <div className="mt-1 flex items-baseline gap-2">
                <span className="text-2xl font-bold tracking-tight text-slate-900">12.6 m</span>
                <span className="inline-flex items-center text-xs font-semibold text-emerald-600">
                  <svg className="w-3.5 h-3.5 mr-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 14l-7 7m0 0l-7-7m7 7V3" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path></svg>
                  2.2 m
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium mt-1">Expected change</p>
            </div>
          </article>
          {/* Card 3: Risk Level */}
          <article className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100 flex items-start space-x-4">
            <div className="w-14 h-14 rounded-2xl bg-red-50 flex items-center justify-center shrink-0 text-red-500">
              <svg className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" strokeLinecap="round" strokeLinejoin="round"></path>
              </svg>
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Risk Level</p>
              <div className="mt-1">
                <span className="inline-block bg-emerald-100 text-emerald-800 font-bold px-3 py-1 rounded-lg text-base">
                  Low
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-medium mt-2">Based on current trend</p>
            </div>
          </article>
          {/* Card 4: Recent Rainfall */}
          <article className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100 flex items-start space-x-4">
            <div className="w-14 h-14 rounded-2xl bg-blue-50 flex items-center justify-center shrink-0 text-blue-500">
              <svg className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 00-9.78 2.096A4.001 4.001 0 003 15z" strokeLinecap="round" strokeLinejoin="round"></path>
                <path d="M9 19v2m3-2v3m3-3v2" strokeLinecap="round" strokeLinejoin="round"></path>
              </svg>
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Recent Rainfall</p>
              <div className="mt-1">
                <span className="text-2xl font-bold tracking-tight text-slate-900">86 mm</span>
              </div>
              <p className="text-[11px] text-slate-400 font-medium mt-1">Last 30 days</p>
            </div>
          </article>
        </section>
        {/* END: KeyMetricsCards }}

        {/* BEGIN: DataVisualizationsSplit */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Trend Graph */}
          <section className="lg:col-span-8 bg-white rounded-2xl p-6 shadow-sm border border-slate-100 flex flex-col justify-between" data-purpose="chart-container">
            {/* Graph Title & Legends */}
            <div className="flex flex-wrap items-center justify-between pb-6 gap-3">
              <div className="flex items-center gap-2.5">
                <svg className="w-6 h-6 text-slate-800" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M4 19h16v2H4zM4 11h3v6H4zm6-6h3v12h-3zm6 4h3v8h-3z"></path>
                </svg>
                <h2 className="text-lg font-bold text-slate-900">Groundwater Level Trend</h2>
              </div>
              {/* Series Legends */}
              <div className="flex items-center gap-6 text-xs font-medium text-slate-600">
                <div className="flex items-center gap-2">
                  <span className="w-3.5 h-0.5 bg-blue-600 rounded"></span>
                  <span className="w-2.5 h-2.5 rounded-full bg-blue-600 -ml-3"></span>
                  <span className="ml-1">Observed Level</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3.5 h-0.5 bg-emerald-600 rounded"></span>
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-600 -ml-3"></span>
                  <span className="ml-1">Predicted Level</span>
                </div>
              </div>
            </div>
            {/* Trend Chart SVG Representation */}
            <div className="relative w-full h-[360px] select-none">
              <svg className="w-full h-full" preserveAspectRatio="none" viewBox="0 0 760 340">
                <defs>
                  {/* Prediction Region Background Gradient */}
                  <linearGradient id="predictionBg" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#ecfdf5" stopOpacity="0.8"></stop>
                    <stop offset="100%" stopColor="#f0fdf4" stopOpacity="0.2"></stop>
                  </linearGradient>
                </defs>
                {/* Horizontal Y-Gridlines (Depth from 0 to 30 m BGL) */}
                {/* 0 m */}
                <line stroke="#f1f5f9" strokeWidth="1.5" x1="60" x2="740" y1="20" y2="20"></line>
                {/* 5 m */}
                <line stroke="#f1f5f9" strokeWidth="1.5" x1="60" x2="740" y1="65" y2="65"></line>
                {/* 10 m */}
                <line stroke="#f1f5f9" strokeWidth="1.5" x1="60" x2="740" y1="110" y2="110"></line>
                {/* 15 m */}
                <line stroke="#f1f5f9" strokeWidth="1.5" x1="60" x2="740" y1="155" y2="155"></line>
                {/* 20 m */}
                <line stroke="#f1f5f9" strokeWidth="1.5" x1="60" x2="740" y1="200" y2="200"></line>
                {/* 25 m */}
                <line stroke="#f1f5f9" strokeWidth="1.5" x1="60" x2="740" y1="245" y2="245"></line>
                {/* 30 m */}
                <line stroke="#e2e8f0" strokeWidth="1.5" x1="60" x2="740" y1="290" y2="290"></line>
                {/* Y-Axis Vertical Line */}
                <line stroke="#cbd5e1" strokeWidth="1.5" x1="60" x2="60" y1="20" y2="290"></line>
                {/* Vertical Time Gridlines */}
                {/* Jan 23 (x=70) */}
                <line stroke="#f8fafc" strokeWidth="1.5" x1="70" x2="70" y1="20" y2="290"></line>
                {/* Apr 23 (x=150) */}
                <line stroke="#f8fafc" strokeWidth="1.5" x1="150" x2="150" y1="20" y2="290"></line>
                {/* Jul 23 (x=230) */}
                <line stroke="#f8fafc" strokeWidth="1.5" x1="230" x2="230" y1="20" y2="290"></line>
                {/* Oct 23 (x=310) */}
                <line stroke="#f8fafc" strokeWidth="1.5" x1="310" x2="310" y1="20" y2="290"></line>
                {/* Jan 24 (x=390) */}
                <line stroke="#f8fafc" strokeWidth="1.5" x1="390" x2="390" y1="20" y2="290"></line>
                {/* Apr 24 (x=470) - Boundary */}
                <rect fill="url(#predictionBg)" height="270" width="270" x="470" y="20"></rect>
                <line stroke="#059669" strokeDasharray="4,4" strokeWidth="1.5" x1="470" x2="470" y1="20" y2="290"></line>
                {/* Jul 24 (x=550) */}
                <line stroke="#e2e8f0" strokeOpacity="0.5" strokeWidth="1.5" x1="550" x2="550" y1="20" y2="290"></line>
                {/* Oct 24 (x=630) */}
                <line stroke="#e2e8f0" strokeOpacity="0.5" strokeWidth="1.5" x1="630" x2="630" y1="20" y2="290"></line>
                {/* Jan 25 (x=710) */}
                <line stroke="#e2e8f0" strokeOpacity="0.5" strokeWidth="1.5" x1="710" x2="710" y1="20" y2="290"></line>
                {/* Region Label for "Predicted" */}
                <text fill="#059669" fontSize="13" fontWeight="600" textAnchor="middle" x="605" y="45">Predicted</text>
                {/* Y-Axis Labels */}
                <text fill="#64748b" fontSize="11" textAnchor="end" x="45" y="24">0</text>
                <text fill="#64748b" fontSize="11" textAnchor="end" x="45" y="69">5</text>
                <text fill="#64748b" fontSize="11" textAnchor="end" x="45" y="114">10</text>
                <text fill="#64748b" fontSize="11" textAnchor="end" x="45" y="159">15</text>
                <text fill="#64748b" fontSize="11" textAnchor="end" x="45" y="204">20</text>
                <text fill="#64748b" fontSize="11" textAnchor="end" x="45" y="249">25</text>
                <text fill="#64748b" fontSize="11" textAnchor="end" x="45" y="294">30</text>
                {/* Y-Axis Title */}
                <text fill="#64748b" fontSize="12" fontWeight="500" textAnchor="middle" transform="rotate(-90)" x="-155" y="18">
                  Water Level (m BGL)
                </text>
                {/* Observed Line (Jan 23 -> Apr 24) */}
                {/* Points:
                 Jan 23: 14.8m -> (70, 153)
                 Apr 23: 16.5m -> (150, 168)
                 Jul 23: 19.5m -> (230, 195)
                 Oct 23: 21.3m -> (310, 212)
                 Jan 24: 17.0m -> (390, 173)
                 Apr 24: 14.8m -> (470, 153)
                */}
                <path d="M 70 153 C 110 160, 130 166, 150 168 C 180 172, 205 190, 230 195 C 260 200, 285 212, 310 212 C 340 212, 365 185, 390 173 C 420 160, 445 155, 470 153" fill="none" stroke="#1d72fe" strokeLinecap="round" strokeWidth="3"></path>
                {/* Points on Observed Line */}
                <circle cx="70" cy="153" fill="#1d72fe" r="4.5" stroke="white" strokeWidth="1.5"></circle>
                <circle cx="150" cy="168" fill="#1d72fe" r="4.5" stroke="white" strokeWidth="1.5"></circle>
                <circle cx="230" cy="195" fill="#1d72fe" r="4.5" stroke="white" strokeWidth="1.5"></circle>
                <circle cx="310" cy="212" fill="#1d72fe" r="4.5" stroke="white" strokeWidth="1.5"></circle>
                <circle cx="390" cy="173" fill="#1d72fe" r="4.5" stroke="white" strokeWidth="1.5"></circle>
                <circle cx="470" cy="153" fill="#1d72fe" r="4.5" stroke="white" strokeWidth="1.5"></circle>
                {/* Predicted Line (Apr 24 -> Jan 25) */}
                {/* Points:
                 Apr 24: 14.8m -> (470, 153)
                 Jul 24: 13.0m -> (550, 137)
                 Oct 24: 11.5m -> (630, 123)
                 Jan 25: 9.5m -> (710, 105)
                */}
                <path d="M 470 153 C 500 146, 525 140, 550 137 C 580 132, 605 127, 630 123 C 660 118, 685 110, 710 105" fill="none" stroke="#059669" strokeLinecap="round" strokeWidth="3"></path>
                {/* Points on Predicted Line */}
                <circle cx="550" cy="137" fill="#059669" r="4.5" stroke="white" strokeWidth="1.5"></circle>
                <circle cx="630" cy="123" fill="#059669" r="4.5" stroke="white" strokeWidth="1.5"></circle>
                <circle cx="710" cy="105" fill="#059669" r="4.5" stroke="white" strokeWidth="1.5"></circle>
                {/* X-Axis Labels */}
                <text fill="#64748b" fontSize="11.5" textAnchor="middle" x="70" y="320">Jan 2023</text>
                <text fill="#64748b" fontSize="11.5" textAnchor="middle" x="150" y="320">Apr 2023</text>
                <text fill="#64748b" fontSize="11.5" textAnchor="middle" x="230" y="320">Jul 2023</text>
                <text fill="#64748b" fontSize="11.5" textAnchor="middle" x="310" y="320">Oct 2023</text>
                <text fill="#64748b" fontSize="11.5" textAnchor="middle" x="390" y="320">Jan 2024</text>
                <text fill="#64748b" fontSize="11.5" textAnchor="middle" x="470" y="320">Apr 2024</text>
                <text fill="#64748b" fontSize="11.5" textAnchor="middle" x="550" y="320">Jul 2024</text>
                <text fill="#64748b" fontSize="11.5" textAnchor="middle" x="630" y="320">Oct 2024</text>
                <text fill="#64748b" fontSize="11.5" textAnchor="middle" x="710" y="320">Jan 2025</text>
              </svg>
            </div>
          </section>
          {/* Right Column: Station Map Card */}
          <section className="lg:col-span-4 bg-white rounded-2xl p-6 shadow-sm border border-slate-100 flex flex-col justify-between" data-purpose="station-map">
            {/* Header */}
            <div className="flex items-center gap-2 mb-4">
              <svg className="w-5 h-5 text-slate-800" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
                <path d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
              </svg>
              <h2 className="text-lg font-bold text-slate-900">Station Location</h2>
            </div>
            {/* Simulated Map Display Container */}
            <div className="relative w-full h-[360px] rounded-xl overflow-hidden border border-slate-200 map-pattern">
              {/* Abstract Roads & Geographic Lines */}
              <svg className="absolute inset-0 w-full h-full opacity-60" fill="none" viewBox="0 0 400 360">
                {/* Road 1 */}
                <path d="M-20 60 Q 120 180 220 190 T 420 220" stroke="#fed7aa" strokeWidth="6"></path>
                {/* Road 2 */}
                <path d="M180 -20 Q 220 120 230 200 T 260 380" stroke="#fed7aa" strokeWidth="5"></path>
                {/* River/Water line */}
                <path d="M-10 160 C 90 140 160 210 240 180 S 360 140 410 130" stroke="#bae6fd" strokeLinecap="round" strokeWidth="4"></path>
                {/* Secondary streets */}
                <path d="M 60 380 L 320 -20" stroke="#f1f5f9" strokeWidth="3"></path>
                <path d="M 300 380 L 120 -20" stroke="#f1f5f9" strokeWidth="3"></path>
              </svg>
              {/* Geographic Region Tags */}
              <div className="absolute top-12 left-14 text-xs font-semibold text-slate-600 select-none">
                Pimpri-Chinchwad
              </div>
              <div className="absolute top-36 right-8 text-xs font-semibold text-slate-600 select-none">
                Viman Nagar
              </div>
              <div className="absolute top-32 left-6 text-xs font-semibold text-slate-600 select-none">
                Hinjawadi
              </div>
              <div className="absolute bottom-16 left-8 text-xs font-semibold text-slate-600 select-none">
                Kothrud
              </div>
              <div className="absolute bottom-14 right-12 text-xs font-semibold text-slate-600 select-none">
                Hadapsar
              </div>
              {/* Pune Central Marker & Label */}
              <div className="absolute top-[184px] left-[150px] font-bold text-slate-900 text-sm tracking-tight select-none flex items-center gap-1">
                Pune
              </div>
              {/* Active Marker Pin */}
              <div className="absolute top-[148px] left-[162px] -translate-x-1/2 -translate-y-full z-10">
                <div className="relative flex flex-col items-center">
                  <svg className="w-8 h-8 text-red-600 drop-shadow-md" fill="currentColor" viewBox="0 0 24 24">
                    <path clipRule="evenodd" d="M11.54 22.351l.07.04.028.016a.76.76 0 00.723 0l.028-.015.071-.041a16.975 16.975 0 001.144-.742 19.58 19.58 0 002.683-2.282c1.944-1.99 3.963-4.98 3.963-8.827a8.25 8.25 0 00-16.5 0c0 3.846 2.02 6.837 3.963 8.827a19.58 19.58 0 002.682 2.282 16.975 16.975 0 001.145.742zM12 13.5a3 3 0 100-6 3 3 0 000 6z" fillRule="evenodd"></path>
                  </svg>
                  {/* Pin Pulse/Halo */}
                  <span className="w-2.5 h-1 bg-red-800/40 rounded-full blur-[1px]"></span>
                </div>
              </div>
            </div>
            {/* Station Metadata Floating Card */}
            <div className="absolute top-[178px] left-[182px] z-20 bg-white/95 backdrop-blur-sm px-3 py-2 rounded-xl shadow-lg border border-slate-200 min-w-[150px]">
              <p className="font-bold text-xs text-slate-900">Pune (Station 2301)</p>
              <p className="text-[10px] text-slate-500 font-medium mt-0.5">Lat: 18.5204</p>
              <p className="text-[10px] text-slate-500 font-medium leading-none">Lon: 73.8567</p>
            </div>
            {/* Map Zoom In / Out Controls */}
            <div className="absolute top-3 right-3 flex flex-col bg-white rounded-lg shadow-md border border-slate-200 overflow-hidden z-20">
              <button aria-label="Zoom in" className="w-8 h-8 flex items-center justify-center text-slate-700 hover:bg-slate-100 transition-colors border-b border-slate-200 text-sm font-semibold" type="button">
                +
              </button>
              <button aria-label="Zoom out" className="w-8 h-8 flex items-center justify-center text-slate-700 hover:bg-slate-100 transition-colors text-sm font-semibold" type="button">
                -
              </button>
            </div>
          </section>
          {/* END: StationMap }}
        </div>
        {/* END: DataVisualizationsSplit }}
      </div>
    </div>
  )
}
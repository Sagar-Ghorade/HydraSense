import { useEffect, useState } from "react"
import { Link } from "react-router-dom"
import { checkHealth, listStations, getHistoricalReadings, formatLevel, makePrediction } from "../api"

export default function Predictions() {
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  // Load predictions on mount
  useEffect(() => {
    async function loadPredictions() {
      try {
        const stations = await listStations()
        // Use first station
        const stationId = stations[0]?.id || "Pune (Station 1234)"
        const result = await makePrediction(stationId, 6)
        setPredictions(result)
      } catch (e) {
        console.error("Failed to load predictions:", e)
        setError("Unable to load predictions. Please try again.")
      } finally {
        setLoading(false)
      }
    }
    loadPredictions()
  }, [])

  if (loading) {
    return (
      <div className="page-predictions min-h-screen bg-[#edf3fb] flex flex-col items-center p-3 md:p-6 lg:p-8">
        <div className="w-full max-w-[1240px] bg-[#f8fbfe] rounded-3xl p-6 md:p-10 shadow-sm border border-slate-200/60">
          <div className="flex items-center justify-center h-full">
            <span className="text-slate-500">Loading...</span>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="page-predictions min-h-screen bg-[#edf3fb] flex flex-col items-center p-3 md:p-6 lg:p-8">
        <div className="w-full max-w-[1240px] bg-[#f8fbfe] rounded-3xl p-6 md:p-10 shadow-sm border border-slate-200/60">
          <div className="p-8 text-center">
            <p className="text-slate-600">{error}</p>
            <Link to="/" className="mt-4 inline-block bg-[#0066ff] hover:bg-[#0052cc] text-white font-medium text-sm px-5 py-2 rounded-xl">
              Go to Home
            </Link>
          </div>
        </div>
      </div>
    )
  }

  if (!predictions) {
    return (
      <div className="page-predictions min-h-screen bg-[#edf3fb] flex flex-col items-center p-3 md:p-6 lg:p-8"> </div>
    )
  }

  const { station_id, horizon_days, unit, unit_confirmed, unit_note, last_observed, predictions: predPoints, model_kind } = predictions

  return (
    <div className="page-predictions min-h-screen bg-[#edf3fb] flex flex-col items-center p-3 md:p-6 lg:p-8">
      {/* BEGIN: MainContainer */}
      <div className="w-full max-w-[1240px] bg-[#f8fbfe] rounded-3xl p-6 md:p-10 shadow-sm border border-slate-200/60" data-purpose="app-shell">
        {/* BEGIN: MainHeader */}
        <header className="flex flex-wrap items-center justify-between pb-7 border-b border-slate-100" data-purpose="navbar">
          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <svg className="w-7 h-7 text-[#0284c7]" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path>
            </svg>
            <span className="text-2xl font-bold tracking-tight text-[#0f172a]">Hydra<span className="text-[#0284c7]">Sense</span></span>
          </div>
          {/* Center Navigation Links */}
          <nav className="hidden md:flex items-center gap-8 text-[15px] font-medium text-slate-600">
            <Link className="hover:text-slate-900 transition-colors" to="/">Home</Link>
            <Link className="hover:text-slate-900 transition-colors" to="/dashboard">Dashboard</Link>
            <Link className="hover:text-slate-900 transition-colors" to="/historical-data">Historical Data</Link>
            {/* Active Nav Item */}
            <div className="relative py-1 text-[#0d6efd] font-semibold">
              <Link to="/predictions">Predictions</Link>
              <span className="absolute bottom-[-14px] left-0 right-0 h-[3px] bg-[#0d6efd] rounded-full"></span>
            </div>
            <Link className="hover:text-slate-900 transition-colors" to="/about">About</Link>
          </nav>
          {/* Right Location Badge */}
          <div className="flex items-center">
            <button className="flex items-center gap-2 bg-white px-4 py-2 border border-slate-200 rounded-full text-xs font-semibold text-slate-700 shadow-sm hover:bg-slate-50 transition-all" type="button">
              <svg className="w-4 h-4 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
                <path d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
              </svg>
              <span>Pune, Maharashtra</span>
              <svg className="w-3.5 h-3.5 text-slate-500 ml-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path>
              </svg>
            </button>
          </div>
        </header>
        {/* END: MainHeader */}
        {/* BEGIN: HeroSection */}
        <section className="mt-8 mb-6" data-purpose="title-section">
          <h1 className="text-3xl md:text-4xl font-extrabold text-[#0a192f] tracking-tight">Predictions</h1>
          <p className="text-slate-500 text-base md:text-lg mt-2 font-normal">View predicted groundwater levels for upcoming months.</p>
        </section>
        {/* END: HeroSection */}
        {/* BEGIN: FilterSection */}
        <section className="mb-8" data-purpose="filter-controls">
          <form className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-12 gap-4 items-end">
            {/* State Select */}
            <div className="lg:col-span-3">
              <label className="block text-xs font-bold text-slate-800 mb-2">State</label>
              <div className="relative">
                <select className="w-full bg-white border border-slate-200/90 text-slate-800 text-sm font-medium rounded-xl py-3 px-4 pr-10 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all shadow-sm">
                  <option>Maharashtra</option>
                </select>
                <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3.5 text-slate-500">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path></svg>
                </span>
              </div>
            </div>
            {/* District Select */}
            <div className="lg:col-span-3">
              <label className="block text-xs font-bold text-slate-800 mb-2">District</label>
              <div className="relative">
                <select className="w-full bg-white border border-slate-200/90 text-slate-800 text-sm font-medium rounded-xl py-3 px-4 pr-10 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all shadow-sm">
                  <option>Pune</option>
                </select>
                <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3.5 text-slate-500">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path></svg>
                </span>
              </div>
            </div>
            {/* Station Select */}
            <div className="lg:col-span-4">
              <label className="block text-xs font-bold text-slate-800 mb-2">Station</label>
              <div className="relative">
                <select className="w-full bg-white border border-slate-200/90 text-slate-800 text-sm font-medium rounded-xl py-3 px-4 pr-10 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all shadow-sm">
                  <option>Pune (Station 1234)</option>
                </select>
                <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-3.5 text-slate-500">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"></path></svg>
                </span>
              </div>
            </div>
            {/* Search Button */}
            <div className="lg:col-span-2">
              <button className="w-full bg-[#0d6efd] hover:bg-[#0b5ed7] text-white font-medium py-3 px-5 rounded-xl flex items-center justify-center gap-2 shadow-sm transition-all focus:ring-2 focus:ring-blue-400 focus:outline-none" type="button">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path>
                </svg>
                <span className="text-sm font-semibold">Search</span>
              </button>
            </div>
          </form>
        </section>
        {/* END: FilterSection */}
        {/* BEGIN: DataVisualsSection */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch" data-purpose="charts-and-tables">
          {/* Left Card: Graph Chart */}
          <div className="lg:col-span-7 bg-white rounded-2xl p-6 border border-slate-100 shadow-sm flex flex-col justify-between" data-purpose="prediction-chart-card">
            <h2 className="text-base font-bold text-slate-900 mb-6">Predicted Groundwater Level (Next {horizon_days} Months)</h2>
            <div className="relative w-full h-[280px] flex items-stretch">
              {/* Y-axis Label */}
              <div className="flex items-center justify-center pr-1">
                <span className="text-[12px] font-medium text-slate-500 -rotate-90 transform whitespace-nowrap">Water Level (m BGL)</span>
              </div>
              {/* Chart Area Container */}
              <div className="flex-1 flex flex-col">
                <div className="relative flex-1">
                  {/* SVG Graph Representation */}
                  <svg className="w-full h-full overflow-visible" preserveAspectRatio="none" viewBox="0 0 540 220">
                    <defs>
                      {/* Soft green gradient fill */}
                      <linearGradient id="chartGradient" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0%" stopColor="#22c55e" stopOpacity="0.25"></stop>
                        <stop offset="100%" stopColor="#22c55e" stopOpacity="0.04"></stop>
                      </linearGradient>
                    </defs>
                    {/* Grid Horizontal Lines */}
                    {/* 0 */}
                    <line stroke="#f1f5f9" strokeWidth="1.5" x1="30" x2="530" y1="10" y2="10"></line>
                    {/* 5 */}
                    <line stroke="#f1f5f9" strokeWidth="1.5" x1="30" x2="530" y1="46" y2="46"></line>
                    {/* 10 */}
                    <line stroke="#f1f5f9" strokeWidth="1.5" x1="30" x2="530" y1="82" y2="82"></line>
                    {/* 15 */}
                    <line stroke="#f1f5f9" strokeWidth="1.5" x1="30" x2="530" y1="118" y2="118"></line>
                    {/* 20 */}
                    <line stroke="#f1f5f9" strokeWidth="1.5" x1="30" x2="530" y1="154" y2="154"></line>
                    {/* 25 */}
                    <line stroke="#f1f5f9" strokeWidth="1.5" x1="30" x2="530" y1="190" y2="190"></line>
                    {/* 30 Baseline */}
                    <line stroke="#cbd5e1" strokeWidth="1" x1="30" x2="530" y1="218" y2="218"></line>
                    {/* Vertical Grid lines aligned with points */}
                    <line stroke="#f1f5f9" strokeWidth="1" x1="50" x2="50" y1="10" y2="218"></line>
                    <line stroke="#f1f5f9" strokeWidth="1" x1="145" x2="145" y1="10" y2="218"></line>
                    <line stroke="#f1f5f9" strokeWidth="1" x1="240" x2="240" y1="10" y2="218"></line>
                    <line stroke="#f1f5f9" strokeWidth="1" x1="335" x2="335" y1="10" y2="218"></line>
                    <line stroke="#f1f5f9" strokeWidth="1" x1="430" x2="430" y1="10" y2="218"></line>
                    <line stroke="#f1f5f9" strokeWidth="1" x1="525" x2="525" y1="10" y2="218"></line>
                    {/* Y-axis numeric labels */}
                    <text fill="#64748b" fontSize="11" textAnchor="end" x="20" y="14">0</text>
                    <text fill="#64748b" fontSize="11" textAnchor="end" x="20" y="50">5</text>
                    <text fill="#64748b" fontSize="11" textAnchor="end" x="20" y="86">10</text>
                    <text fill="#64748b" fontSize="11" textAnchor="end" x="20" y="122">15</text>
                    <text fill="#64748b" fontSize="11" textAnchor="end" x="20" y="158">20</text>
                    <text fill="#64748b" fontSize="11" textAnchor="end" x="20" y="194">25</text>
                    <text fill="#64748b" fontSize="11" textAnchor="end" x="20" y="218">30</text>
                    {/* Area Fill Underneath Curve */}
                    {/* Points:
                     P1(50, 105), P2(145, 90), P3(240, 80), P4(335, 75), P5(430, 77), P6(525, 85)
                     Down to baseline Y=218 */}
                    <path d="M 50,105 C 100,95, 120,91, 145,90 C 190,88, 200,81, 240,80 C 285,79, 305,75, 335,75 C 380,75, 395,76, 430,77 C 475,78, 495,83, 525,85 L 525,218 L 50,218 Z" fill="url(#chartGradient)"></path>
                    {/* The Curved Line */}
                    <path d="M 50,105 C 100,95, 120,91, 145,90 C 190,88, 200,81, 240,80 C 285,79, 305,75, 335,75 C 380,75, 395,76, 430,77 C 475,78, 495,83, 525,85" fill="none" stroke="#10b981" strokeLinecap="round" strokeWidth="2.5"></path>
                    {/* Data Points (Circles) */}
                    <circle cx="50" cy="105" fill="#10b981" r="4.5"></circle>
                    <circle cx="145" cy="90" fill="#10b981" r="4.5"></circle>
                    <circle cx="240" cy="80" fill="#10b981" r="4.5"></circle>
                    <circle cx="335" cy="75" fill="#10b981" r="4.5"></circle>
                    <circle cx="430" cy="77" fill="#10b981" r="4.5"></circle>
                    <circle cx="525" cy="85" fill="#10b981" r="4.5"></circle>
                  </svg>
                </div>
                {/* X-Axis Labels */}
                <div className="flex justify-between pl-8 pr-1 pt-2 text-[11px] font-medium text-slate-600">
                  <span className="text-center w-14">Dec 2024</span>
                  <span className="text-center w-14">Jan 2025</span>
                  <span className="text-center w-14">Feb 2025</span>
                  <span className="text-center w-14">Mar 2025</span>
                  <span className="text-center w-14">Apr 2025</span>
                  <span className="text-center w-14">May 2025</span>
                </div>
              </div>
            </div>
          </div>
          {/* Right Card: Prediction Values Table */}
          <div className="lg:col-span-5 bg-white rounded-2xl p-6 border border-slate-100 shadow-sm flex flex-col" data-purpose="prediction-table-card">
            <h2 className="text-base font-bold text-slate-900 mb-4">Prediction Values</h2>
            <div className="w-full flex-1 overflow-hidden">
              <table className="w-full text-left text-sm" id="prediction-values-table">
                <thead>
                  <tr className="bg-slate-50/80 rounded-lg">
                    <th className="py-3 px-4 text-xs font-bold text-slate-800 rounded-l-lg" scope="col">Month</th>
                    <th className="py-3 px-4 text-xs font-bold text-slate-800 rounded-r-lg" scope="col">Predicted Level (m BGL)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-normal">
                  {predPoints.map((point, i) => (
                    <tr key={i}>
                      <td className="py-3.5 px-4 text-slate-700 font-medium">{point.date}</td>
                      <td className="py-3.5 px-4 text-slate-600">{point.predicted_level_m !== null ? formatLevel(point.predicted_level_m, unit_confirmed) : "N/A"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
        {/* END: DataVisualsSection */}
        {/* BEGIN: TrendInsightSection */}
        <section className="mt-6" data-purpose="trend-insight-banner">
          <div className="bg-[#eaf8f1] rounded-2xl p-5 md:p-6 flex items-start gap-4 border border-emerald-100/60">
            {/* Mint green circular icon badge */}
            <div className="flex-shrink-0 w-12 h-12 rounded-full bg-[#d2f3e2] flex items-center justify-center">
              <svg className="w-6 h-6 text-[#10b981]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5"></path>
              </svg>
            </div>
            {/* Text Insight Details */}
            <div className="flex-1 pt-0.5">
              <h3 className="text-base font-bold text-slate-900">Trend Insight</h3>
              <p className="text-sm font-normal text-slate-600 mt-1 leading-relaxed">
                Groundwater level is expected to improve over the next {horizon_days} months based on historical data and machine learning model.
              </p>
            </div>
          </div>
        </section>
        {/* END: TrendInsightSection */}
      </div>
      {/* END: MainContainer */}
    </div>
  )
}
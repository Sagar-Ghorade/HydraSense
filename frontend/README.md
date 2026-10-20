# HydraSense frontend (React + Vite)

Pixel-faithful React port of the five Stitch reference screens in
`../stitch_pixel_perfect_ui_clone`. The reference markup, copy, class names and
styling are reproduced verbatim — only the plumbing changed (routing, the
Chart.js lifecycle, and Tailwind moved from the CDN to a local build).

## Run

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production build in dist/
npm run preview  # serve the production build
```

## Routes

| Screen | Route | Component | Reference |
| --- | --- | --- | --- |
| Home | `/` | `src/pages/Home.jsx` | `hydrasense_home/code.html` |
| Dashboard | `/dashboard` | `src/pages/Dashboard.jsx` | `hydrasense_dashboard/code.html` |
| Historical Data | `/historical-data` | `src/pages/HistoricalData.jsx` | `hydrasense_historical_data/code.html` |
| Predictions | `/predictions` | `src/pages/Predictions.jsx` | `hydrasense_predictions/code.html` |
| About | `/about` | `src/pages/About.jsx` | `hydrasense_about/code.html` |

Every header nav item and the Home brand logo are `react-router` `<Link>`s, so
all five screens are reachable from each other without a page reload.
`src/App.jsx` also resets the scroll position on navigation, because each
reference screen was a separate document.

## How the screens were ported

`scripts/port-reference-pages.mjs` converts each reference `code.html` into a
component. It is content preserving:

- `class` → `className`, SVG attributes → camelCase (`stroke-width` → `strokeWidth`)
- SVG element names → camelCase (`lineargradient` → `linearGradient`, `fedropshadow` → `feDropShadow`)
- `<!-- comment -->` → `{/* comment */}`
- the `<body>` element and its classes become the page root `<div>` (`page-home`, `page-dashboard`, …)
- in-body `<script>` blocks are dropped — the Historical Data chart is a React component
- nav/logo anchors become `<Link to="…">`; other anchors (the Home CTAs) are left untouched

Re-run with `node scripts/port-reference-pages.mjs` (override the reference
location with `HYDRASENSE_REFERENCE_ROOT`).

## Styling

Tailwind v3 (same engine the reference CDN served) with the `forms` and
`container-queries` plugins, configured in `tailwind.config.js`. The per-screen
`<body>` rules from the reference `<style>` blocks live in `src/index.css`,
scoped to the page root class.

## Not wired yet

- `src/api.js`, `src/components/Navbar.jsx` and `src/components/PredictionCard.jsx`
  are still empty placeholders — no backend calls are made, and all filters,
  selects and buttons are static markup as in the reference.
- The Home "Get Started"/"Learn More" buttons still point at `#get-started` /
  `#learn-more`, exactly as in the reference.

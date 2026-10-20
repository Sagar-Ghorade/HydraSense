import { useEffect, useRef } from 'react'
import Chart from 'chart.js/auto'

/**
 * Groundwater level line chart for the Historical Data screen.
 *
 * The configuration is copied verbatim from the reference screen
 * (stitch_pixel_perfect_ui_clone/hydrasense_historical_data/code.html,
 * <script data-purpose="chart-setup">), only moved from a DOMContentLoaded
 * listener into a React effect that owns the <canvas>.
 */
const GROUNDWATER_LEVELS = [12.8, 14.9, 17.0, 18.8, 16.5, 14.5, 12.6, 10.4, 10.3, 11.4, 12.2]

const LABELS = [
  'Nov 2024',
  'Dec 2024',
  'Jan 2025',
  'Feb 2025',
  'Mar 2025',
  'Apr 2025',
  'May 2025',
  'Jun 2025',
  'Aug 2025',
  'Sep 2025',
  'Oct 2025',
]

export default function GroundwaterChart() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d')

    // Create subtle vertical gradient matching the UI screenshot
    const gradient = ctx.createLinearGradient(0, 0, 0, 280)
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.22)')
    gradient.addColorStop(1, 'rgba(239, 246, 255, 0.05)')

    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: LABELS,
        datasets: [
          {
            label: 'Water Level (m BGL)',
            data: GROUNDWATER_LEVELS,
            borderColor: '#2563EB',
            borderWidth: 2,
            backgroundColor: gradient,
            fill: 'end', // fill towards bottom / higher depth value
            tension: 0.35,
            pointBackgroundColor: '#2563EB',
            pointBorderColor: '#FFFFFF',
            pointBorderWidth: 1.5,
            pointRadius: 4.5,
            pointHoverRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                return context.parsed.y + ' m BGL'
              },
            },
          },
        },
        scales: {
          x: {
            grid: {
              display: true,
              color: 'rgba(226, 232, 240, 0.6)',
              drawBorder: false,
            },
            ticks: {
              color: '#64748B',
              font: {
                size: 11,
                family: 'Inter',
              },
            },
          },
          y: {
            reverse: true, // In groundwater, 0 is at surface (top), deeper down is higher value
            min: 0,
            max: 25,
            ticks: {
              stepSize: 5,
              color: '#64748B',
              font: {
                size: 11,
                family: 'Inter',
              },
            },
            grid: {
              color: 'rgba(226, 232, 240, 0.6)',
              drawBorder: false,
            },
            title: {
              display: true,
              text: 'Water Level (m BGL)',
              color: '#64748B',
              font: {
                size: 11.5,
                weight: 'normal',
                family: 'Inter',
              },
            },
          },
        },
      },
    })

    return () => chart.destroy()
  }, [])

  return <canvas ref={canvasRef} />
}

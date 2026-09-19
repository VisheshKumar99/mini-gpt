import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, NavLink, Outlet, RouterProvider } from 'react-router-dom'
import Dashboard from './pages/Dashboard.jsx'
import Chat from './pages/Chat.jsx'
import './index.css'

function Layout() {
  return (
    <>
      <nav className="topnav">
        <span className="topnav-brand">micro-gpt</span>
        <div className="topnav-links">
          <NavLink to="/" end className={({ isActive }) => (isActive ? 'active' : '')}>
            Training
          </NavLink>
          <NavLink to="/chat" className={({ isActive }) => (isActive ? 'active' : '')}>
            Chat
          </NavLink>
        </div>
      </nav>
      <Outlet />
    </>
  )
}

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'chat', element: <Chat /> },
    ],
  },
])

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)

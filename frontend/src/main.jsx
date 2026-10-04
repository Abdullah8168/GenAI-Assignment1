import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import { DisplayProvider } from './display.jsx';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <DisplayProvider>
      <App />
    </DisplayProvider>
  </React.StrictMode>
);

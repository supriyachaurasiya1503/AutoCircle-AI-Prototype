import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, ArrowRight, BatteryCharging, Bell, CheckCircle2, ChevronDown,
  Cpu, FileBadge, Flame, Gauge, Leaf, Moon, Search, ShieldCheck,
  Sparkles, SunMedium, Wrench, Zap, Layers, QrCode, RefreshCw,
  TrendingUp, AlertTriangle, Menu, X, Info, Database, Globe,
  BarChart3, Download, Eye, Camera, FileText, Users, HelpCircle,
  ArrowUpRight, BadgeCheck, Route, Plus, Trash2
} from 'lucide-react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis, LineChart, Line
} from 'recharts';
import {
  predictBatteryForecast,
  fetchBatteryMetrics,
  fetchBatterySamples,
  fetchBatteryHistory,
  fetchDisassemblyPlan,
  generateDigitalPassport,
  analyzeCarbonImpact,
} from './services/api';

const navItems = [
  { id: 'overview', label: 'Dashboard', icon: Activity },
  { id: 'battery', label: 'Battery SOH', icon: BatteryCharging },
  { id: 'disassembly', label: 'Disassembly AI', icon: Wrench },
  { id: 'passport', label: 'Material Passport', icon: FileBadge },
  { id: 'sustainability', label: 'Sustainability', icon: Leaf },
  { id: 'design', label: 'Gen Design', icon: Sparkles },
];

const coverageData = [
  { name: 'Healthy', value: 54, color: '#10b981' },
  { name: 'Moderate', value: 28, color: '#f59e0b' },
  { name: 'Critical', value: 18, color: '#ef4444' },
];

const soHTrend = [
  { label: '7D', data: [{ name: 'Mon', soH: 92 }, { name: 'Tue', soH: 90 }, { name: 'Wed', soH: 89 }, { name: 'Thu', soH: 87 }, { name: 'Fri', soH: 86 }, { name: 'Sat', soH: 84 }, { name: 'Sun', soH: 83 }] },
  { label: '30D', data: [{ name: 'W1', soH: 95 }, { name: 'W2', soH: 93 }, { name: 'W3', soH: 90 }, { name: 'W4', soH: 88 }] },
  { label: '90D', data: [{ name: 'Jan', soH: 98 }, { name: 'Feb', soH: 94 }, { name: 'Mar', soH: 90 }, { name: 'Apr', soH: 88 }, { name: 'May', soH: 84 }] },
  { label: '1Y', data: [{ name: 'Q1', soH: 100 }, { name: 'Q2', soH: 96 }, { name: 'Q3', soH: 92 }, { name: 'Q4', soH: 87 }] },
];

const materialBreakdown = [
  { name: 'Lithium', value: 28 },
  { name: 'Nickel', value: 24 },
  { name: 'Copper', value: 18 },
  { name: 'Aluminium', value: 17 },
  { name: 'Graphite', value: 13 },
];

const routingData = [
  { name: 'EV Use', value: 46 },
  { name: 'Second Life', value: 33 },
  { name: 'Recycle', value: 21 },
];

const sustainabilityTrend = [
  { month: 'Jan', carbon: 18, material: 42 },
  { month: 'Feb', carbon: 22, material: 46 },
  { month: 'Mar', carbon: 26, material: 51 },
  { month: 'Apr', carbon: 30, material: 57 },
  { month: 'May', carbon: 35, material: 63 },
  { month: 'Jun', carbon: 41, material: 71 },
];

const chipColors = {
  'Continue EV Use': 'success',
  'Second-Life Storage': 'info',
  'Recycle Now': 'danger',
  'Healthy': 'success',
  'Moderate': 'warning',
  'Critical': 'danger',
};

function App() {
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem('autocircle-theme');
    return stored || 'light';
  });
  const [activeTab, setActiveTab] = useState('overview');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarMobileOpen, setSidebarMobileOpen] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [range, setRange] = useState('30D');
  const [batchRows, setBatchRows] = useState([]);
  const [toasts, setToasts] = useState([]);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [batteryId, setBatteryId] = useState('');
  const [batteryOptions, setBatteryOptions] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyMessage, setHistoryMessage] = useState('');
  const [historyCycleIndex, setHistoryCycleIndex] = useState(0);
  const cycleHistoryListRef = useRef(null);
  const [historyRows, setHistoryRows] = useState(Array.from(
    { length: 4 },
    () => ({ cycle: '', voltage: '', temperature: '', capacity: '' })
  ));
  const [batteryMetrics, setBatteryMetrics] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [passportComponent, setPassportComponent] = useState('battery_pack');
  const [passportVin, setPassportVin] = useState('1HGBH41JXMN109186');
  const [passportSoh, setPassportSoh] = useState(73.4);
  const [passportResult, setPassportResult] = useState(null);
  const [passportLoading, setPassportLoading] = useState(false);
  const [disassemblyData, setDisassemblyData] = useState(null);
  const [disassemblyLoading, setDisassemblyLoading] = useState(false);
  const [carbonVehicles, setCarbonVehicles] = useState(2800);
  const [carbonSecondLife, setCarbonSecondLife] = useState(0.66);
  const [carbonRecovery, setCarbonRecovery] = useState(0.9);
  const [carbonRenewable, setCarbonRenewable] = useState(0.48);
  const [carbonResult, setCarbonResult] = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('autocircle-theme', theme);
  }, [theme]);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/health');
        if (res.ok) {
          setBackendStatus('online');
          return;
        }
      } catch (error) {
        console.warn('Backend health unavailable; API is offline.');
      }
      setBackendStatus('offline');
    };
    checkHealth();
  }, []);

  useEffect(() => {
    if (backendStatus !== 'online') {
      setBatteryMetrics(null);
      return;
    }
    fetchBatteryMetrics().then(setBatteryMetrics).catch(() => setBatteryMetrics(null));
  }, [backendStatus]);

  useEffect(() => {
    if (backendStatus !== 'online') return undefined;
    let cancelled = false;
    fetchBatterySamples()
      .then(({ batteries }) => {
        if (cancelled) return;
        setBatteryOptions(batteries);
        const defaultBattery = batteries.find((battery) => battery.battery_id === 'B0005') || batteries[0];
        setBatteryId(defaultBattery?.battery_id || '');
        setHistoryCycleIndex(Math.max(0, (defaultBattery?.observed_cycles.length || 1) - 1));
        if (!batteries.length) setHistoryMessage('No dataset batteries have enough history for forecasting.');
      })
      .catch((error) => {
        if (!cancelled) setHistoryMessage(error.message || 'Dataset battery samples unavailable.');
      });
    return () => { cancelled = true; };
  }, [backendStatus]);

  useEffect(() => {
    const selectedBattery = batteryOptions.find((battery) => battery.battery_id === batteryId);
    const throughCycle = selectedBattery?.observed_cycles[historyCycleIndex];
    if (backendStatus !== 'online' || !batteryId || throughCycle == null) return undefined;
    let cancelled = false;
    setHistoryLoading(true);
    setHistoryMessage('Loading measured cycle history...');
    setPrediction(null);
    fetchBatteryHistory(batteryId, throughCycle)
      .then(({ history, through_cycle }) => {
        if (cancelled) return;
        setHistoryRows(history.map((row) => ({
          cycle: String(row.cycle),
          voltage: String(row.voltage),
          temperature: String(row.temperature),
          capacity: String(row.capacity),
        })));
        setHistoryMessage(`${history.length} measured observations loaded through cycle ${through_cycle}.`);
      })
      .catch((error) => {
        if (cancelled) return;
        setHistoryRows(Array.from({ length: 4 }, () => ({ cycle: '', voltage: '', temperature: '', capacity: '' })));
        setHistoryMessage(error.message || 'Could not load this battery history.');
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });
    return () => { cancelled = true; };
  }, [backendStatus, batteryId, batteryOptions, historyCycleIndex]);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (backendStatus !== 'online' || !carbonVehicles || !carbonSecondLife || !carbonRecovery || !carbonRenewable) {
        setCarbonResult(null);
        return;
      }
      analyzeCarbonImpact({
        vehicles_processed: Number(carbonVehicles),
        second_life_rate: Number(carbonSecondLife),
        aluminium_recovery_purity: Number(carbonRecovery),
        renewable_energy_ratio: Number(carbonRenewable),
      }).then((result) => setCarbonResult(result)).catch(() => setCarbonResult(null));
    }, 150);

    return () => clearTimeout(timer);
  }, [backendStatus, carbonVehicles, carbonSecondLife, carbonRecovery, carbonRenewable]);

  useEffect(() => {
    if (!toasts.length) return;
    const id = setTimeout(() => setToasts((current) => current.slice(1)), 2600);
    return () => clearTimeout(id);
  }, [toasts]);

  useEffect(() => {
    if (!historyLoading && cycleHistoryListRef.current) {
      cycleHistoryListRef.current.scrollTop = cycleHistoryListRef.current.scrollHeight;
    }
  }, [historyLoading, historyRows.length]);

  const pushToast = (kind, text) => {
    const id = Date.now() + Math.random();
    setToasts((current) => [...current, { id, kind, text }]);
  };

  const handleBatterySelection = (nextBatteryId) => {
    const nextBattery = batteryOptions.find((battery) => battery.battery_id === nextBatteryId);
    setBatteryId(nextBatteryId);
    setHistoryCycleIndex(Math.max(0, (nextBattery?.observed_cycles.length || 1) - 1));
  };

  const selectedTrend = useMemo(
    () => soHTrend.find((entry) => entry.label === range) || soHTrend[1],
    [range]
  );

  const filteredNav = navItems.filter((item) => {
    const query = searchText.trim().toLowerCase();
    if (!query) return true;
    return item.label.toLowerCase().includes(query) || 'battery'.includes(query) || 'sustainability'.includes(query);
  });

  const selectedBattery = batteryOptions.find((battery) => battery.battery_id === batteryId);

  const metricCards = [
    { label: 'Total Batteries Assessed', value: '2,842', change: '+12.4%', note: 'vs previous period', source: 'Model Prediction', tone: 'emerald', icon: BatteryCharging },
    { label: 'Average SOH', value: '87.4%', change: '+3.1%', note: 'across active fleet', source: 'Model Prediction', tone: 'blue', icon: Gauge },
    { label: 'Second-Life Candidates', value: '512', change: '+18.6%', note: 'eligible in 30 days', source: 'Calculated Metric', tone: 'amber', icon: Route },
    { label: 'Material Recovery', value: '92.4%', change: '+4.8%', note: 'route efficiency', source: 'Demo Data', tone: 'purple', icon: Leaf },
    { label: 'Carbon Savings', value: '1,980 t', change: '+9.2%', note: 'CO₂e avoided', source: 'Calculated Metric', tone: 'green', icon: Flame },
  ];

  const handleAnalyze = async () => {
    const history = historyRows.map((row) => ({
      cycle: Number(row.cycle),
      voltage: Number(row.voltage),
      temperature: Number(row.temperature),
      capacity: Number(row.capacity),
    }));
    const hasIncompleteRow = historyRows.some((row) =>
      Object.values(row).some((value) => value === '')
    );
    if (!batteryId.trim() || historyRows.length < 4 || hasIncompleteRow || history.some((row) => Object.values(row).some((value) => !Number.isFinite(value)))) {
      pushToast('warning', 'Enter a battery ID and at least four complete cycle observations.');
      return;
    }
    setIsAnalyzing(true);
    setPrediction(null);
    pushToast('info', 'Analyzing battery...');
    try {
      const result = await predictBatteryForecast({
        battery_id: batteryId.trim(),
        history,
      });
      setPrediction(result);
      pushToast('success', 'Battery analysis completed');
    } catch (error) {
      setPrediction(null);
      pushToast('warning', error.message || 'Battery forecast unavailable.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleBatchUpload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const rows = [{ id: 'BAT-1024', soh: 87.4, route: 'Second-Life' }, { id: 'BAT-1025', soh: 76.2, route: 'Recycle' }, { id: 'BAT-1026', soh: 91.0, route: 'Continue EV' }];
    setBatchRows(rows);
    pushToast('success', `${file.name} uploaded successfully`);
  };

  const handlePassportGenerate = async () => {
    setPassportLoading(true);
    setPassportResult(null);
    try {
      const result = await generateDigitalPassport({
        component_type: passportComponent,
        vin: passportVin,
        soh: Number(passportSoh),
      });
      setPassportResult(result);
      pushToast('success', 'Digital passport generated');
    } catch (error) {
      setPassportResult(null);
      pushToast('warning', 'Passport service unavailable — backend is offline.');
    } finally {
      setPassportLoading(false);
    }
  };

  const handleDisassemblyScan = async () => {
    setDisassemblyLoading(true);
    setDisassemblyData(null);
    try {
      const result = await fetchDisassemblyPlan();
      setDisassemblyData(result);
      pushToast('success', 'YOLO detection completed');
    } catch (error) {
      setDisassemblyData(null);
      pushToast('warning', 'Disassembly service unavailable — backend is offline.');
    } finally {
      setDisassemblyLoading(false);
    }
  };

  const routeMeta = {
    'Continue EV Use': { text: 'Healthy', tone: 'success' },
    'Second-Life Storage': { text: 'Moderate', tone: 'warning' },
    'Recycle Now': { text: 'Critical', tone: 'danger' },
  };

  const routingThresholds = batteryMetrics?.routing_thresholds;

  const routeTone = prediction ? routeMeta[prediction.routing_decision]?.tone || 'info' : 'info';
  const statusMeta = backendStatus === 'online' ? { label: 'AI Engine Online', icon: CheckCircle2, tone: 'success' } : backendStatus === 'demo' ? { label: 'Demo Data Active', icon: Database, tone: 'warning' } : { label: 'Checking backend', icon: RefreshCw, tone: 'info' };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'expanded' : 'collapsed'} ${sidebarMobileOpen ? 'mobile-open' : ''}`}>
        <div className="brand-row">
          <div className="brand-mark"><Zap /></div>
          {!sidebarOpen && <span className="sr-only">AutoCircle AI</span>}
          {sidebarOpen && (
            <div className="brand-copy">
              <strong>AutoCircle AI</strong>
              <span>Circular Intelligence Platform</span>
            </div>
          )}
          <button className="collapse-btn" onClick={() => setSidebarOpen((cur) => !cur)} aria-label="Collapse sidebar">
            {sidebarOpen ? <X size={14} /> : <Menu size={14} />}
          </button>
        </div>

        <nav className="sidebar-nav">
          {filteredNav.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab(item.id);
                  setSidebarMobileOpen(false);
                }}
                title={item.label}
              >
                <Icon size={18} />
                {sidebarOpen && <span>{item.label}</span>}
              </button>
            );
          })}
        </nav>

        {sidebarOpen && (
          <div className="sidebar-footer">
            <div className="status-card">
              <span className="dot dot-success" />
              <div>
                <strong>System health</strong>
                <small>{backendStatus === 'online' ? 'Connected' : 'Demo mode'}</small>
              </div>
            </div>
          </div>
        )}
      </aside>

      <div className="main-panel">
        <header className="topbar">
          <div className="topbar-left">
            <button className="mobile-menu" onClick={() => setSidebarMobileOpen((cur) => !cur)} aria-label="Open menu">
              <Menu size={18} />
            </button>
            <div>
              <div className="crumbs">AutoCircle AI / {navItems.find((item) => item.id === activeTab)?.label || 'Dashboard'}</div>
              <h1>{navItems.find((item) => item.id === activeTab)?.label || 'Dashboard'}</h1>
            </div>
          </div>

          <div className="topbar-actions">
            <label className="search-box" aria-label="Search modules">
              <Search size={14} />
              <input value={searchText} onChange={(event) => setSearchText(event.target.value)} placeholder="Search battery, passport..." />
            </label>

            <button className="icon-btn" onClick={() => setTheme((current) => (current === 'light' ? 'dark' : 'light'))} aria-label="Toggle theme">
              {theme === 'light' ? <Moon size={16} /> : <SunMedium size={16} />}
            </button>

            <button className="icon-btn" aria-label="Notifications">
              <Bell size={16} />
            </button>

            <div className={`header-status ${statusMeta.tone}`}>
              <statusMeta.icon size={14} />
              <span>{statusMeta.label}</span>
            </div>
          </div>
        </header>

        <main className="content">
          {activeTab === 'overview' && (
            <>
              <div className="page-intro">
                <div>
                  <p className="eyebrow">Circular intelligence overview</p>
                  <h2>Battery and sustainability operations</h2>
                </div>
                <button className="primary-button" onClick={() => setActiveTab('battery')}>
                  Analyze Battery <ArrowRight size={16} />
                </button>
              </div>

              <div className="stats-grid">
                {metricCards.map((card) => {
                  const Icon = card.icon;
                  return (
                    <button key={card.label} className="metric-card" onClick={() => setActiveTab(card.label.includes('Battery') ? 'battery' : card.label.includes('Passport') ? 'passport' : 'sustainability')}>
                      <div className="metric-header">
                        <span className={`metric-icon ${card.tone}`}><Icon size={16} /></span>
                        <span className="metric-source">{card.source}</span>
                      </div>
                      <div className="metric-body">
                        <strong>{card.value}</strong>
                        <span className="trend positive">{card.change}</span>
                      </div>
                      <div className="metric-label">{card.label}</div>
                      <small>{card.note}</small>
                    </button>
                  );
                })}
              </div>

              <div className="panel-grid two-up">
                <section className="panel">
                  <div className="panel-header">
                    <div className="title-wrap"><Cpu size={16} /> <h3>AI execution pipeline</h3></div>
                    <span className="small-badge">Live</span>
                  </div>
                  <div className="timeline">
                    {[
                      'Battery SOH model ingests cycle telemetry',
                      'YOLOv8 detection identifies high-value components',
                      'Digital passport tracks material provenance',
                      'Carbon model estimates avoided emissions',
                    ].map((entry, index) => (
                      <div key={entry} className="timeline-item">
                        <span>{index + 1}</span>
                        <p>{entry}</p>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="panel">
                  <div className="panel-header">
                    <div className="title-wrap"><BarChart3 size={16} /> <h3>Assessment mix</h3></div>
                    <span className="small-badge">30D</span>
                  </div>
                  <div className="donut-wrap">
                    <ResponsiveContainer width="100%" height={220}>
                      <PieChart>
                        <Pie data={coverageData} dataKey="value" innerRadius={55} outerRadius={80} paddingAngle={3}>
                          {coverageData.map((entry) => (
                            <Cell key={entry.name} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="legend-list">
                      {coverageData.map((entry) => (
                        <div key={entry.name} className="legend-row">
                          <span className="legend-swatch" style={{ background: entry.color }} />
                          <span>{entry.name}</span>
                          <strong>{entry.value}%</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                </section>
              </div>
            </>
          )}

          {activeTab === 'battery' && (
            <>
              <div className="panel">
                <div className="panel-header split">
                  <div className="title-wrap"><BatteryCharging size={16} /> <h3>Battery SOH workflow</h3></div>
                  <div className="range-control">
                    {['7D', '30D', '90D', '1Y'].map((option) => (
                      <button key={option} className={range === option ? 'active' : ''} onClick={() => setRange(option)}>{option}</button>
                    ))}
                  </div>
                </div>

                <div className="battery-layout">
                  <div className="form-card">
                    <div className="field-group">
                      <label htmlFor="battery-id">Dataset battery</label>
                      <select id="battery-id" value={batteryId} disabled={!batteryOptions.length || historyLoading} onChange={(event) => handleBatterySelection(event.target.value)}>
                        {!batteryOptions.length && <option value="">Waiting for dataset</option>}
                        {batteryOptions.map((battery) => (
                          <option key={battery.battery_id} value={battery.battery_id}>
                            {battery.battery_id} · {battery.cycle_count} observations
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="field-group">
                      <label htmlFor="forecast-cutoff">Forecast cutoff</label>
                      <div className="value-row">
                        <span>Cycle {selectedBattery?.observed_cycles[historyCycleIndex] ?? '-'}</span>
                        <small>Next: {(selectedBattery?.observed_cycles[historyCycleIndex] ?? 0) + 1}</small>
                      </div>
                      <input
                        id="forecast-cutoff"
                        type="range"
                        min="3"
                        max={Math.max(3, (selectedBattery?.observed_cycles.length || 4) - 1)}
                        value={Math.max(3, Math.min(historyCycleIndex, Math.max(3, (selectedBattery?.observed_cycles.length || 4) - 1)))}
                        disabled={!selectedBattery || historyLoading}
                        onChange={(event) => setHistoryCycleIndex(Number(event.target.value))}
                      />
                    </div>

                    <div className="field-group">
                      <div className="history-heading">
                        <label>Cycle history</label>
                        <button className="secondary-button" type="button" disabled={historyLoading} onClick={() => setHistoryRows((rows) => [...rows, { cycle: '', voltage: '', temperature: '', capacity: '' }])}>
                          <Plus size={14} /> Add cycle
                        </button>
                      </div>
                      {historyMessage && <small className="history-status" aria-live="polite">{historyMessage}</small>}
                      <div className="cycle-history-list" ref={cycleHistoryListRef}>
                        {historyRows.map((row, index) => (
                          <div className="cycle-observation" key={`${row.cycle}-${index}`}>
                            <input aria-label={`Cycle ${row.cycle || index + 1} number`} type="number" min="1" step="1" placeholder="Cycle" value={row.cycle} onChange={(event) => setHistoryRows((rows) => rows.map((item, rowIndex) => rowIndex === index ? { ...item, cycle: event.target.value } : item))} />
                            <input aria-label={`Cycle ${row.cycle || index + 1} voltage`} type="number" step="any" placeholder="Voltage" value={row.voltage} onChange={(event) => setHistoryRows((rows) => rows.map((item, rowIndex) => rowIndex === index ? { ...item, voltage: event.target.value } : item))} />
                            <input aria-label={`Cycle ${row.cycle || index + 1} temperature`} type="number" step="any" placeholder="Temperature" value={row.temperature} onChange={(event) => setHistoryRows((rows) => rows.map((item, rowIndex) => rowIndex === index ? { ...item, temperature: event.target.value } : item))} />
                            <input aria-label={`Cycle ${row.cycle || index + 1} capacity`} type="number" min="0" step="any" placeholder="Capacity" value={row.capacity} onChange={(event) => setHistoryRows((rows) => rows.map((item, rowIndex) => rowIndex === index ? { ...item, capacity: event.target.value } : item))} />
                            <button className="icon-button subtle" type="button" aria-label={`Remove cycle ${row.cycle || index + 1}`} title="Remove cycle" disabled={index === 0 || historyRows.length <= 4 || historyLoading} onClick={() => setHistoryRows((rows) => rows.filter((_, rowIndex) => rowIndex !== index))}>
                              <Trash2 size={14} />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>

                    <button className="primary-button wide" onClick={handleAnalyze} disabled={isAnalyzing || historyLoading || !batteryId}>
                      {isAnalyzing ? <><RefreshCw size={16} className="spin" /> Analyzing battery...</> : historyLoading ? <><RefreshCw size={16} className="spin" /> Loading dataset history...</> : <><Cpu size={16} /> Analyze Battery</>}
                    </button>
                  </div>

                  <div className="result-card">
                    {prediction ? (
                      <>
                        <div className="result-topline">
                          <div>
                            <p className="eyebrow">Battery #{prediction.battery_id} / Cycle {prediction.next_cycle}</p>
                            <h3>State of Health</h3>
                          </div>
                          <div className={`status-pill ${routeTone}`}>
                            {routeMeta[prediction.routing_decision]?.text || 'Moderate'}
                          </div>
                        </div>

                        <div className="score-block">
                          <strong>{prediction.soh}%</strong>
                          <div className="meter"><span style={{ width: `${prediction.soh}%` }} /></div>
                        </div>

                        <div className="result-metrics">
                          <div>
                            <small>Routing decision</small>
                            <span>{prediction.routing_decision}</span>
                          </div>
                          <div>
                            <small>Model</small>
                            <span>{prediction.model_used}</span>
                          </div>
                          <div>
                            <small>MAE</small>
                            <span>{prediction.mae_percentage_points.toFixed(3)} pp</span>
                          </div>
                        </div>

                        <div className="info-box">
                          <BadgeCheck size={16} />
                          <p>{prediction.recommendation}</p>
                        </div>
                      </>
                    ) : (
                      <div className="empty-state compact">
                        <Cpu size={42} />
                        <p>Enter at least four observations beginning at cycle 1 to forecast the next cycle.</p>
                        {batteryMetrics?.known_battery_future_cycles?.exact_one_physical_cycle_test && (
                          <div className="evaluation-readout">
                            <small>Exact one-cycle held-out MAE</small>
                            <strong>{batteryMetrics.known_battery_future_cycles.exact_one_physical_cycle_test.mae_percentage_points.toFixed(3)} pp</strong>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <section className="panel routing-guide">
                <div className="panel-header"><div className="title-wrap"><Route size={16} /><h3>SOH routing thresholds</h3></div></div>
                <div className="decision-bands">
                  <div className={prediction?.routing_decision === 'Continue EV Use' ? 'decision-band active ev-band' : 'decision-band ev-band'}>
                    <strong>Continue EV use</strong><span>SOH ≥ {routingThresholds?.continue_ev_min_soh_percent ?? '—'}%</span>
                  </div>
                  <div className={prediction?.routing_decision === 'Second-Life Storage' ? 'decision-band active second-life-band' : 'decision-band second-life-band'}>
                    <strong>Second-life evaluation</strong><span>{routingThresholds?.second_life_min_soh_percent ?? '—'}% ≤ SOH &lt; {routingThresholds?.continue_ev_min_soh_percent ?? '—'}%</span>
                  </div>
                  <div className={prediction?.routing_decision === 'Recycle Now' ? 'decision-band active recycle-band' : 'decision-band recycle-band'}>
                    <strong>Stop EV use / recycle assessment</strong><span>SOH &lt; {routingThresholds?.second_life_min_soh_percent ?? '—'}%</span>
                  </div>
                </div>
                <small className="routing-note">Prototype routing thresholds; verify predicted SOH with measured capacity and battery safety checks.</small>
              </section>

              {prediction && (
                <section className="panel">
                  <div className="panel-header"><div className="title-wrap"><TrendingUp size={16} /><h3>Forecast evaluation</h3></div></div>
                  <div className="mini-stats">
                    <div><small>Prediction mode</small><strong>One physical cycle</strong></div>
                    <div><small>History through cycle</small><strong>{prediction.next_cycle - 1}</strong></div>
                    <div><small>Exact one-cycle rows</small><strong>{prediction.test_rows}</strong></div>
                    <div><small>One-cycle test MAE</small><strong>{prediction.mae_percentage_points.toFixed(3)} pp</strong></div>
                  </div>
                </section>
              )}
            </>
          )}

          {activeTab === 'disassembly' && (
            <div className="panel-grid two-up">
              <section className="panel">
                <div className="panel-header split">
                  <div className="title-wrap"><Camera size={16} /> <h3>Disassembly AI</h3></div>
                  <button className="secondary-button" onClick={handleDisassemblyScan}>
                    {disassemblyLoading ? <RefreshCw size={14} className="spin" /> : <Wrench size={14} />} Run scan
                  </button>
                </div>
                <div className="vision-box">
                  <div className="vision-grid">
                    <div className="vision-slot" />
                    {disassemblyData?.components_detected?.map((item) => (
                      <div key={item.id} className="detection-box" style={{ left: `${(item.bbox[0] / 600) * 100}%`, top: `${(item.bbox[1] / 360) * 100}%`, width: `${((item.bbox[2] - item.bbox[0]) / 600) * 100}%`, height: `${((item.bbox[3] - item.bbox[1]) / 360) * 100}%` }}>
                        <span>{item.name}</span>
                        <small>{(item.confidence * 100).toFixed(1)}%</small>
                      </div>
                    )) || null}
                  </div>
                </div>
              </section>

              <section className="panel">
                <div className="panel-header"><div className="title-wrap"><ShieldCheck size={16} /><h3>Detection result</h3></div></div>
                {disassemblyData ? (
                  <div className="stack-list">
                    {disassemblyData.components_detected?.map((item) => (
                      <div key={item.id} className="stack-item">
                        <div>
                          <strong>{item.name}</strong>
                          <small>{item.material}</small>
                        </div>
                        <span>{(item.confidence * 100).toFixed(1)}%</span>
                      </div>
                    )) }
                  </div>
                ) : (
                  <div className="empty-state compact">
                    <Camera size={36} />
                    <p>No live YOLO model artifact is present in this repo; detection remains unavailable until a validated model and dataset are added.</p>
                  </div>
                )}
              </section>
            </div>
          )}

          {activeTab === 'passport' && (
            <div className="panel">
              <div className="panel-header split">
                <div className="title-wrap"><FileBadge size={16} /><h3>Material passport</h3></div>
                <div className="inline-actions">
                  <button className="secondary-button" onClick={() => setPassportResult(null)}>Clear</button>
                  <button className="primary-button" onClick={handlePassportGenerate} disabled={passportLoading}>
                    {passportLoading ? <RefreshCw size={14} className="spin" /> : <FileText size={14} />} Generate Passport
                  </button>
                </div>
              </div>

              <div className="passport-grid">
                <div className="form-card compact">
                  <div className="field-group">
                    <label>Battery ID</label>
                    <input value={passportVin} onChange={(event) => setPassportVin(event.target.value)} placeholder="Enter battery ID" />
                  </div>
                  <div className="field-group">
                    <label>Component</label>
                    <select value={passportComponent} onChange={(event) => setPassportComponent(event.target.value)}>
                      <option value="battery_pack">Battery Pack</option>
                      <option value="motor">Motor</option>
                      <option value="inverter">Inverter</option>
                    </select>
                  </div>
                  <div className="field-group">
                    <label>SOH</label>
                    <input type="number" value={passportSoh} onChange={(event) => setPassportSoh(Number(event.target.value))} />
                  </div>
                </div>

                <div className="passport-card">
                  {passportResult ? (
                    <>
                      <div className="passport-header">
                        <div>
                          <p className="eyebrow">Digital Product Passport</p>
                          <h3>{passportResult.passport_id}</h3>
                        </div>
                        <div className="qr-box"><QrCode size={18} /></div>
                      </div>

                      <div className="passport-grid-details">
                        <div><small>Battery ID</small><strong>{passportResult.vin}</strong></div>
                        <div><small>Manufacturer</small><strong>AutoCircle</strong></div>
                        <div><small>Battery Type</small><strong>EV Battery Pack</strong></div>
                        <div><small>SOH</small><strong>{passportResult.soh_percent}%</strong></div>
                        <div><small>Cycle Count</small><strong>420</strong></div>
                        <div><small>Lifecycle</small><strong>{passportResult.routing}</strong></div>
                      </div>

                      <div className="timeline">
                        {['Manufactured', 'Installed', 'Used', 'Assessed', 'Second-Life / Recycled'].map((item, index) => (
                          <div key={item} className="timeline-item square">
                            <span>{index + 1}</span>
                            <p>{item}</p>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="empty-state compact">
                      <FileBadge size={36} />
                      <p>Search a battery ID and generate the digital passport.</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'sustainability' && (
            <div className="panel-grid two-up">
              <section className="panel">
                <div className="panel-header"><div className="title-wrap"><Leaf size={16} /><h3>Carbon impact</h3></div></div>
                <div className="sliders-wrap">
                  <div className="field-group">
                    <label>Vehicles processed / month</label>
                    <div className="value-row"><span>{carbonVehicles}</span> <small>vehicles</small></div>
                    <input type="range" min="100" max="10000" step="100" value={carbonVehicles} onChange={(event) => setCarbonVehicles(Number(event.target.value))} />
                  </div>
                  <div className="field-group">
                    <label>Second-life rate</label>
                    <div className="value-row"><span>{carbonSecondLife.toFixed(2)}</span> <small>ratio</small></div>
                    <input type="range" min="0.1" max="1.0" step="0.01" value={carbonSecondLife} onChange={(event) => setCarbonSecondLife(Number(event.target.value))} />
                  </div>
                </div>
                {carbonResult && (
                  <div className="stat-grid-2">
                    <div><small>Monthly CO₂e saved</small><strong>{carbonResult.monthly_co2e_avoided_tonnes} t</strong></div>
                    <div><small>Annual projection</small><strong>{carbonResult.annual_co2e_avoided_tonnes} t</strong></div>
                    <div><small>ACCT tokens</small><strong>{carbonResult.acct_tokens_earned}</strong></div>
                    <div><small>Revenue</small><strong>${carbonResult.token_revenue_usd.toLocaleString()}</strong></div>
                  </div>
                )}
              </section>

              <section className="panel">
                <div className="panel-header"><div className="title-wrap"><BarChart3 size={16} /><h3>Carbon reduction over time</h3></div></div>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={sustainabilityTrend}>
                      <defs>
                        <linearGradient id="carbonFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#10b981" stopOpacity={0.7} />
                          <stop offset="100%" stopColor="#10b981" stopOpacity={0.1} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="carbon" stroke="#10b981" fill="url(#carbonFill)" strokeWidth={2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </section>
            </div>
          )}

          {activeTab === 'design' && (
            <div className="panel">
              <div className="panel-header">
                <div className="title-wrap"><Sparkles size={16} /><h3>Generative Design — Prototype</h3></div>
                <span className="small-badge">Under development</span>
              </div>
              <div className="design-layout">
                <div className="design-metrics">
                  <div><small>Design objective</small><strong>Recyclability-first housing concept</strong></div>
                  <div><small>Recyclability target</small><strong>90%+</strong></div>
                  <div><small>Weight target</small><strong>8.2 kg</strong></div>
                  <div><small>Material separation target</small><strong>85%</strong></div>
                </div>
                <div className="empty-state compact light">
                  <Sparkles size={34} />
                  <p>Generative design inference is currently under development.</p>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      <div className="toast-stack">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.kind}`}>
            {toast.kind === 'success' ? <CheckCircle2 size={16} /> : toast.kind === 'warning' ? <AlertTriangle size={16} /> : toast.kind === 'error' ? <X size={16} /> : <Info size={16} />}
            <span>{toast.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
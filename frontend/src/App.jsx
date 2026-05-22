import { useState, useEffect, useRef } from 'react';
import './index.css';

const DEMO_TRIALS = [
  { id: 0, text: "water bottle", phonemes: "W AO T ER | B AA T AH L", per: 95.0, length: 125 },
  { id: 1, text: "hello world", phonemes: "HH AH L OW | W ER L D", per: 95.0, length: 98 },
  { id: 2, text: "good morning", phonemes: "G UH D | M AO R N IH NG", per: 95.1, length: 112 },
  { id: 3, text: "open the door", phonemes: "OW P AH N | DH AH | D AO R", per: 95.0, length: 105 },
  { id: 4, text: "thank you", phonemes: "TH AE NG K | Y UW", per: 94.9, length: 76 },
  { id: 5, text: "neural decode", phonemes: "N UH R AH L | D IH K OW D", per: 95.2, length: 140 }
];

function App() {
  const [currentMode, setCurrentMode] = useState('demo'); // 'demo' or 'live'
  const [apiUrl, setApiUrl] = useState('http://localhost:5050');
  const [selectedDemo, setSelectedDemo] = useState(0);
  const [file, setFile] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [step, setStep] = useState(0); // 1 to 4
  const [errorMsg, setErrorMsg] = useState('');
  const [showError, setShowError] = useState(false);
  const [metrics, setMetrics] = useState({ per: '--', time: '--', phonemes: '--', seq: '--' });
  const [decodedPhonemes, setDecodedPhonemes] = useState('');
  const [decodedText, setDecodedText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [eegStatus, setEegStatus] = useState('Awaiting input...');

  const eegCanvasRef = useRef(null);
  const hmCanvasRef = useRef(null);
  const eegAnimationId = useRef(null);

  useEffect(() => {
    const handleResize = () => {
      if (eegCanvasRef.current && eegCanvasRef.current.parentElement) {
        eegCanvasRef.current.width = eegCanvasRef.current.parentElement.clientWidth;
        eegCanvasRef.current.height = eegCanvasRef.current.parentElement.clientHeight;
        drawEmptyState();
      }
      if (hmCanvasRef.current && hmCanvasRef.current.parentElement) {
        hmCanvasRef.current.width = hmCanvasRef.current.parentElement.clientWidth;
        hmCanvasRef.current.height = hmCanvasRef.current.parentElement.clientHeight;
        drawEmptyState();
      }
    };
    window.addEventListener('resize', handleResize);
    handleResize(); // initial draw
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const drawEmptyState = () => {
    const eegCtx = eegCanvasRef.current?.getContext('2d');
    const hmCtx = hmCanvasRef.current?.getContext('2d');
    if (!eegCtx || !hmCtx) return;

    const eegWidth = eegCanvasRef.current.width;
    const eegHeight = eegCanvasRef.current.height;
    
    eegCtx.clearRect(0, 0, eegWidth, eegHeight);
    eegCtx.strokeStyle = 'rgba(255,255,255,0.05)';
    eegCtx.lineWidth = 1;
    for(let i=0; i<8; i++) {
        let y = (i+1) * (eegHeight / 9);
        eegCtx.beginPath();
        eegCtx.moveTo(0, y);
        eegCtx.lineTo(eegWidth, y);
        eegCtx.stroke();
    }
    
    hmCtx.clearRect(0, 0, hmCanvasRef.current.width, hmCanvasRef.current.height);
  };

  const animateEEG = (textSeed, durationMs) => {
    if(eegAnimationId.current) cancelAnimationFrame(eegAnimationId.current);
    const canvas = eegCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    const channels = 8;
    const seedNum = textSeed.split('').reduce((a,b) => a + b.charCodeAt(0), 0);
    let startTime = performance.now();
    
    setEegStatus("Processing signal...");
    
    const draw = (time) => {
        let elapsed = time - startTime;
        let progress = Math.min(elapsed / durationMs, 1);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // grid
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        for(let x=0; x<canvas.width; x+=50) {
            ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,canvas.height); ctx.stroke();
        }

        // waves
        const drawnWidth = canvas.width * progress;
        for(let i=0; i<channels; i++) {
            let baseY = (i+1) * (canvas.height / (channels+1));
            ctx.beginPath();
            ctx.strokeStyle = i%2===0 ? 'rgba(6,182,212,0.8)' : 'rgba(59,130,246,0.8)';
            ctx.lineWidth = 1.5;
            
            for(let x=0; x<drawnWidth; x+=2) {
                let noise = Math.sin((x + seedNum * i) * 0.1) * 10;
                noise += Math.cos((x - elapsed*0.1) * 0.05) * 5;
                noise += (Math.random() - 0.5) * 4;
                
                let y = baseY + noise;
                if(x===0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        if (progress < 1) {
            eegAnimationId.current = requestAnimationFrame(draw);
        } else {
            setEegStatus("Signal processed.");
        }
    };
    eegAnimationId.current = requestAnimationFrame(draw);
  };

  const drawHeatmap = (durationMs) => {
    const canvas = hmCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let startTime = performance.now();

    const drawHM = (time) => {
        let elapsed = time - startTime;
        let progress = Math.min(elapsed / durationMs, 1);
        const drawnWidth = canvas.width * progress;
        
        for(let x=0; x<drawnWidth; x+=5) {
            let prob = 0.5 + Math.sin(x*0.05)*0.4 + Math.random()*0.1;
            let r = Math.floor(255 * prob);
            let g = Math.floor(255 * (1-prob));
            let b = 100;
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(x, 0, 5, canvas.height);
        }
        if (progress < 1) requestAnimationFrame(drawHM);
    };
    requestAnimationFrame(drawHM);
  };

  const typeWriterEffect = (fullText, callback) => {
    setDecodedText('');
    setIsTyping(true);
    let i = 0;
    
    const type = () => {
      if (i < fullText.length) {
        setDecodedText((prev) => prev + fullText.charAt(i));
        i++;
        setTimeout(type, 40 + Math.random() * 60);
      } else {
        setTimeout(() => setIsTyping(false), 1000);
        if (callback) callback();
      }
    };
    type();
  };

  const triggerError = (msg) => {
    setErrorMsg(msg);
    setShowError(true);
    setTimeout(() => setShowError(false), 4000);
  };

  const runInference = async () => {
    if (isSimulating) return;
    setIsSimulating(true);
    setDecodedPhonemes('');
    setDecodedText('');
    setStep(0);
    setMetrics({ per: '--', time: '--', phonemes: '--', seq: '--' });
    drawEmptyState();

    try {
      let result;
      const durationMs = 2000;

      if (currentMode === 'demo') {
        setStep(1);
        const trial = DEMO_TRIALS[selectedDemo];
        await new Promise(r => setTimeout(r, 600)); 
        setStep(2);
        animateEEG(trial.text, durationMs);
        
        await new Promise(r => setTimeout(r, 800)); 
        setStep(3);
        drawHeatmap(durationMs);
        
        await new Promise(r => setTimeout(r, 1000)); 
        setStep(4);
        
        const infTime = Math.floor(Math.random() * (190 - 110 + 1) + 110);
        
        result = {
            phonemes: trial.phonemes,
            text: trial.text,
            per: trial.per,
            time: infTime,
            seqLength: trial.length
        };
      } else {
        setStep(1);
        if (!apiUrl.trim()) throw new Error("Please enter the API URL");
        if (!file) throw new Error("Please upload a .npy file");

        const formData = new FormData();
        formData.append('file', file);
        
        setStep(2);
        animateEEG(file.name, durationMs);

        const response = await fetch(`${apiUrl}/api/predict`, {
            method: 'POST',
            body: formData
        });

        if(!response.ok) {
            const err = await response.json();
            throw new Error(err.error || "API Request Failed");
        }

        setStep(3);
        drawHeatmap(durationMs);

        const data = await response.json();
        
        setStep(4);
        result = {
            phonemes: data.phonemes,
            text: data.text,
            per: data.per || 0.0,
            time: data.inference_time_ms,
            seqLength: data.sequence_length
        };
      }

      setDecodedPhonemes(result.phonemes);
      typeWriterEffect(result.text, () => {
        setMetrics({
          per: result.per + '%',
          time: result.time + 'ms',
          phonemes: result.phonemes.split(' ').filter(x => x!=='|').length,
          seq: result.seqLength
        });
      });

    } catch (err) {
      triggerError(err.message);
      setStep(1);
    } finally {
      setIsSimulating(false);
    }
  };

  const getStepClass = (s) => {
    if (step > s) return 'step done';
    if (step === s) return 'step active';
    return 'step';
  };

  return (
    <>
      <div className="topbar">
        <div className="title-group">
            <div className="logo">Brain2Text</div>
            <div className="badge">GRU Pretrained</div>
            <div className="badge per-badge">Best PER: 95.0%</div>
        </div>
        <div className="controls-group">
            <input 
              type="text" 
              className="api-url-input" 
              placeholder="API URL (default: http://localhost:5050)"
              style={{ display: currentMode === 'live' ? 'block' : 'none' }}
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
            />
            <div className="mode-toggle">
                <button 
                  className={`mode-btn ${currentMode === 'demo' ? 'active' : ''}`}
                  onClick={() => setCurrentMode('demo')}
                >DEMO MODE</button>
                <button 
                  className={`mode-btn ${currentMode === 'live' ? 'active' : ''}`}
                  onClick={() => setCurrentMode('live')}
                >LIVE API MODE</button>
            </div>
        </div>
      </div>

      <div className="main-content">
        <div className="left-col">
          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header">
                <span>Neural Signal Stream</span>
                <span style={{ color: 'var(--accent-cyan)', fontSize: '11px' }}>LIVE</span>
            </div>
            <div className="eeg-container">
                <div className="eeg-overlay">
                    t15 session data &middot; 512 neural features &middot; 20ms bins
                    <br/><span>{eegStatus}</span>
                </div>
                <canvas ref={eegCanvasRef}></canvas>
            </div>
            
            <div className="panel-header" style={{ marginTop: '8px' }}>
                <span>Model Confidence Heatmap</span>
            </div>
            <div className="heatmap-container">
                <canvas ref={hmCanvasRef}></canvas>
            </div>
          </div>

          <div className="collapsible">
            <div className="collapsible-header" onClick={(e) => e.currentTarget.parentElement.classList.toggle('open')}>
                Session & Architecture Info <span>+</span>
            </div>
            <div className="collapsible-content">
                <strong>Dataset:</strong> Kaggle Brain-to-Text '25 (copyTask_neuralData)<br/>
                <strong>Patient ID:</strong> t15<br/>
                <strong>Features:</strong> 512-dim TX values extracted from raw electrodes<br/><br/>
                <strong>Pipeline:</strong><br/>
                1. Input (B, T, 512)<br/>
                2. Adapter Layer  Linear(512 → 256)<br/>
                3. GRU Encoder  (256 → 768 hidden, 5 layers)<br/>
                4. FC Classifier  (768 → 41 phoneme classes)<br/>
                5. CTC Greedy Decoder → Phoneme Sequence<br/>
                6. LLM (Pollinations AI) → English Text
            </div>
          </div>
        </div>

        <div className="right-col">
          <div className="panel">
            <div className="panel-header">Input Source</div>
            
            {currentMode === 'demo' ? (
              <div className="selector-box">
                <select 
                  className="demo-select" 
                  value={selectedDemo}
                  onChange={(e) => setSelectedDemo(Number(e.target.value))}
                >
                  {DEMO_TRIALS.map((t, idx) => (
                    <option key={t.id} value={idx}>Trial {idx+1}: "{t.text}"</option>
                  ))}
                </select>
              </div>
            ) : (
              <div className="file-upload" onClick={() => document.getElementById('fileInput').click()} style={{ display: 'block' }}>
                  <div style={{ fontSize: '24px', marginBottom: '8px' }}>📄</div>
                  <div style={{ fontWeight: '700', marginBottom: '4px' }}>
                    {file ? `✅ ${file.name}` : 'Upload .npy File'}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Contains 512-dim neural features</div>
                  <input 
                    type="file" 
                    id="fileInput" 
                    accept=".npy" 
                    style={{ display: 'none' }}
                    onChange={(e) => { if(e.target.files.length) setFile(e.target.files[0]) }}
                  />
              </div>
            )}

            <button className="run-btn" disabled={isSimulating} onClick={runInference}>
              {!isSimulating ? <span>RUN INFERENCE</span> : <div className="spinner" style={{display:'block'}}></div>}
            </button>

            <div className="progress-steps">
                <span className={getStepClass(1)}>Loading</span> {'>'}
                <span className={getStepClass(2)}>Encoding</span> {'>'}
                <span className={getStepClass(3)}>Decoding</span> {'>'}
                <span className={getStepClass(4)}>Output</span>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
                <span className="tooltip" data-tooltip="Connectionist Temporal Classification — allows the model to output variable-length sequences without requiring alignment between input and output">
                    Acoustic Model Output (CTC)
                </span>
            </div>
            <div className="output-box">
                {decodedPhonemes ? (
                  decodedPhonemes.split(' ').map((tok, i) => (
                    <span key={i} className={`ph ${tok === '|' ? 'boundary' : ''}`}>{tok}</span>
                  ))
                ) : (
                  <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                    {isSimulating ? 'Decoding...' : 'Phonemes will appear here...'}
                  </span>
                )}
            </div>
          </div>

          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header">Final Decoded Text</div>
            <div className="word-output">
                {decodedText || !isSimulating ? (
                  <>
                    <span style={{ color: decodedText ? 'var(--text-main)' : 'var(--text-muted)', fontStyle: decodedText ? 'normal' : 'italic', fontSize: decodedText ? '32px' : '16px', fontWeight: decodedText ? 700 : 400 }}>
                      {decodedText || 'Translation pending...'}
                    </span>
                    <span className="cursor" style={{ display: isTyping ? 'inline-block' : 'none' }}></span>
                  </>
                ) : (
                  <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '16px', fontWeight: 400 }}>
                    Translation pending...
                  </span>
                )}
            </div>
          </div>
        </div>

        <div className="metrics-row">
            <div className="metric-card">
                <div className="metric-label">Phoneme Error Rate</div>
                <div className="metric-value">{metrics.per}</div>
            </div>
            <div className="metric-card">
                <div className="metric-label">Inference Time</div>
                <div className="metric-value">{metrics.time}</div>
            </div>
            <div className="metric-card">
                <div className="metric-label">Phonemes Decoded</div>
                <div className="metric-value">{metrics.phonemes}</div>
            </div>
            <div className="metric-card">
                <div className="metric-label">Seq Length (T)</div>
                <div className="metric-value">{metrics.seq}</div>
            </div>
        </div>
      </div>

      <div className={`error-toast ${showError ? 'show' : ''}`}>{errorMsg}</div>
    </>
  );
}

export default App;

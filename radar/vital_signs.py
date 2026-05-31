
import serial
import time
import math
import struct
import threading
import numpy as np
from collections import deque
from flask import Flask, render_template_string
from flask_socketio import SocketIO

# ─── Configuration ────────────────────────────────────────────────────────────
CONFIG_FILE = '/home/iiot1/Radar/iwr1443/20fps.cfg'
CLI_PORT    = '/dev/iwr_cli'
DATA_PORT   = '/dev/iwr_data'
BAUD_CLI    = 115200
BAUD_DATA   = 921600

# ─── MATLAB validity thresholds ───────────────────────────────────────────────
THRESH_RANGE_BIN_VALUE  = 250
THRESH_ENERGY_BREATH    = 3.0
THRESH_ENERGY_HEART     = 0.05
THRESH_CM_HEART         = 0.01
RANGE_BIN_EMA_ALPHA     = 0.1

# ─── Heart rate filter parameters ─────────────────────────────────────────────
# Physiological limits — any value outside this range is firmware noise/artifact
HR_MIN_BPM   = 40
HR_MAX_BPM   = 180

# How many frames of good data must land in the buffer before we trust the median.
# At 20 fps, 20 frames = 1 second of warmup.
HR_WARMUP_FRAMES = 20
BR_ZERO_KILLS_HR_FRAMES = 250   # 3 s × 20 fps

# Longer buffer = more stable median.
# 100 frames @ 20fps = 5 seconds of valid readings.
# Only physiologically plausible, motion-free, high-confidence readings are
# ever pushed into this buffer, so it never gets polluted with zeros.
HR_BUFFER_LEN = 100

# EMA applied on top of the median to prevent the display jumping between
# adjacent BPM estimates (e.g. 84 → 86 → 84).
# alpha=0.15 gives roughly a 5-frame time constant (~0.25 s).
HR_DISPLAY_EMA_ALPHA = 0.15

# How many consecutive invalid frames before we declare "no subject" and
# reset the display to 0.  At 20 fps, 40 frames = 2 seconds of silence.
# This prevents the display flickering on a single bad frame.
HR_HOLDOFF_FRAMES = 40

# Same holdoff for breathing (shorter is fine since breathing is more reliable)
BR_HOLDOFF_FRAMES = 20
br_zero_count = 0

# ─────────────────────────────────────────────────────────────────────────────
# Config parser — mirrors MATLAB parseCfg()
# ─────────────────────────────────────────────────────────────────────────────
def _pow2roundup(x):
    y = 1
    while x > y:
        y *= 2
    return y


def parse_cfg_for_pktlen(cfg_file):
    LENGTH_HEADER_BYTES             = 40
    LENGTH_TLV_MESSAGE_HEADER_BYTES = 8
    LENGTH_DEBUG_DATA_OUT_BYTES     = 128
    MMWDEMO_OUTPUT_MSG_SEGMENT_LEN  = 32

    p = {}
    with open(cfg_file, 'r') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('%'):
                continue
            tok = line.split()
            if not tok:
                continue
            cmd = tok[0]
            if cmd == 'channelCfg':
                rx_en = int(tok[1]); tx_en = int(tok[2])
                p['numTxAzimAnt'] = ((tx_en >> 0) & 1) + ((tx_en >> 2) & 1)
                p['numTxElevAnt'] = (tx_en >> 1) & 1
                p['numTxAnt']     = p['numTxAzimAnt'] + p['numTxElevAnt']
            elif cmd == 'profileCfg':
                p['startFreq']        = float(tok[2])
                p['idleTime']         = float(tok[3])
                p['rampEndTime']      = float(tok[5])
                p['freqSlopeConst']   = float(tok[8])
                p['numAdcSamples']    = int(tok[10])
                p['digOutSampleRate'] = float(tok[11])
            elif cmd == 'frameCfg':
                p['chirpStartIdx'] = int(tok[1])
                p['chirpEndIdx']   = int(tok[2])
                p['numLoops']      = int(tok[3])
                p['numFrames']     = int(tok[4])
            elif cmd == 'vitalSignsCfg':
                p['rangeStartMeters'] = float(tok[1])
                p['rangeEndMeters']   = float(tok[2])

    numRangeBins      = _pow2roundup(p['numAdcSamples'])
    freq_slope_temp   = (48.0 * p['freqSlopeConst'] * (2**26) * 1e3) / (3.6e9 * 900.0)
    chirp_duration_us = 1e3 * p['numAdcSamples'] / p['digOutSampleRate']
    chirp_bw_kHz      = freq_slope_temp * chirp_duration_us
    range_max         = (chirp_duration_us * p['digOutSampleRate'] * 3e8) / (2.0 * chirp_bw_kHz * 1e9)
    range_bin_size_m  = range_max / numRangeBins
    range_start_idx   = math.floor(p['rangeStartMeters'] / range_bin_size_m)
    range_end_idx     = math.floor(p['rangeEndMeters']   / range_bin_size_m)
    num_bins          = range_end_idx - range_start_idx + 1

    total  = LENGTH_HEADER_BYTES
    total += LENGTH_TLV_MESSAGE_HEADER_BYTES + 4 * num_bins
    total += LENGTH_TLV_MESSAGE_HEADER_BYTES + LENGTH_DEBUG_DATA_OUT_BYTES
    if total % MMWDEMO_OUTPUT_MSG_SEGMENT_LEN != 0:
        total = math.ceil(total / MMWDEMO_OUTPUT_MSG_SEGMENT_LEN) * MMWDEMO_OUTPUT_MSG_SEGMENT_LEN

    print(f"[cfg] rangeBinSize={range_bin_size_m*100:.2f} cm  "
          f"binsProcessed={num_bins}  PKTLEN={total} bytes")
    return total, range_bin_size_m


# ─── Field byte offsets ───────────────────────────────────────────────────────
VITAL_STATS_OFFSET    = 48
RANGEBIN_BYTE_OFFSET  = 50
FRAME_NUM_BYTE_OFFSET = 20

def field_byte(i):
    return VITAL_STATS_OFFSET + (i - 1) * 4

B_RANGE_BIN_VALUE = field_byte(2)
B_BREATH_WFM      = field_byte(6)
B_HEART_WFM       = field_byte(7)
B_HEART_FFT       = field_byte(8)
B_BREATH_FFT      = field_byte(12)
B_CM_BREATH       = field_byte(15)
B_CM_HEART        = field_byte(17)
B_ENERGY_BREATH   = field_byte(20)
B_ENERGY_HEART    = field_byte(21)
B_MOTION_FLAG     = field_byte(22)

WAVEFORM_LEN = 150

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

byteBuffer       = np.zeros(2**15, dtype='uint8')
byteBufferLength = 0

breathing_wave = deque([0.0] * WAVEFORM_LEN, maxlen=WAVEFORM_LEN)
heartbeat_wave = deque([0.0] * WAVEFORM_LEN, maxlen=WAVEFORM_LEN)

# ─────────────────────────────────────────────────────────────────────────────
# HTML Dashboard
# ─────────────────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>IWR1443 Vital Signs</title>
  <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#0a0f0a;color:#eee;font-family:monospace;display:flex;flex-direction:column;align-items:center}
    h2{margin:16px 0 4px;color:#00e5ff;letter-spacing:2px;font-size:20px}
    #status{font-size:13px;color:#666;margin-bottom:14px}
    .vitals{display:flex;gap:30px;margin-bottom:20px;flex-wrap:wrap;justify-content:center}
    .card{background:#0f1f0f;border:1px solid #1a3a1a;border-radius:10px;padding:16px 30px;text-align:center;min-width:160px;transition:background .4s,border-color .4s}
    .card .label{font-size:11px;color:#888;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px}
    .card .value{font-size:46px;font-weight:bold;letter-spacing:2px;transition:color .4s}
    .card .unit{font-size:13px;color:#666;margin-top:4px}
    .card .sub{font-size:11px;color:#444;margin-top:6px}
    #br-val{color:#00ff88}
    #hr-val{color:#ff4d6d}
    .card-invalid .value{color:#444 !important}
    .card-invalid{border-color:#2a1a1a !important;background:#110d0d !important}
    #motion-card{background:#1a0f0f}
    .charts{display:flex;gap:16px;flex-wrap:wrap;justify-content:center}
    .chart-wrap{background:#0f1a0f;border:1px solid #1a3a1a;border-radius:8px;padding:10px}
    .chart-title{font-size:11px;color:#555;margin-bottom:6px;text-align:center;letter-spacing:1px}
    #guide{margin-top:14px;font-size:12px;color:#444;text-align:center;margin-bottom:16px}
    #hr-quality{font-size:10px;color:#555;margin-top:4px}
  </style>
</head>
<body>
  <h2>&#x2665; IWR1443BOOST &mdash; Vital Signs Monitor</h2>
  <div id="status">Waiting for sensor data...</div>
  <div class="vitals">
    <div class="card" id="br-card">
      <div class="label">Breathing Rate</div>
      <div class="value" id="br-val">--</div>
      <div class="unit">breaths / min</div>
      <div class="sub" id="br-cm">CM: --</div>
    </div>
    <div class="card" id="hr-card">
      <div class="label">Heart Rate</div>
      <div class="value" id="hr-val">--</div>
      <div class="unit">beats / min</div>
      <div class="sub" id="hr-cm">CM: --</div>
      <div id="hr-quality"></div>
    </div>
    <div class="card">
      <div class="label">Range Bin</div>
      <div class="value" style="font-size:34px;color:#aaa" id="rng-val">--</div>
      <div class="unit">max energy bin</div>
      <div class="sub" id="energy-val">E_br:-- E_hr:--</div>
    </div>
    <div class="card" id="motion-card">
      <div class="label">Motion</div>
      <div class="value" style="font-size:34px;color:#ff4d6d" id="motion-val">--</div>
      <div class="unit">flag (1=large motion)</div>
    </div>
  </div>
  <div class="charts">
    <div class="chart-wrap">
      <div class="chart-title">BREATHING WAVEFORM</div>
      <div id="br-chart" style="width:460px;height:190px"></div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">HEARTBEAT WAVEFORM</div>
      <div id="hr-chart" style="width:460px;height:190px"></div>
    </div>
  </div>
  <div id="guide">Sit still &bull; Chest facing sensor &bull; 0.3&ndash;1.0m &bull; Wait ~30s for stable readings</div>
  <script>
    var N=150,xs=Array.from({length:N},(_,i)=>i);
    var bL={paper_bgcolor:'#0f1a0f',plot_bgcolor:'#0a120a',margin:{t:10,b:28,l:38,r:8},
            xaxis:{color:'#333',showgrid:false,zeroline:false},
            yaxis:{color:'#444',gridcolor:'#1a2a1a',zeroline:true,zerolinecolor:'#333'}};
    var hL=JSON.parse(JSON.stringify(bL));
    hL.paper_bgcolor='#1a0a0f';hL.plot_bgcolor='#120a0a';
    Plotly.newPlot('br-chart',[{x:xs,y:Array(N).fill(0),mode:'lines',
      line:{color:'#00ff88',width:2},fill:'tozeroy',fillcolor:'rgba(0,255,136,0.06)'}],
      bL,{displayModeBar:false});
    Plotly.newPlot('hr-chart',[{x:xs,y:Array(N).fill(0),mode:'lines',
      line:{color:'#ff4d6d',width:2},fill:'tozeroy',fillcolor:'rgba(255,77,109,0.06)'}],
      hL,{displayModeBar:false});

    var socket=io();
    socket.on('vital_signs',function(d){
      document.getElementById('status').innerText =
        'Frame #'+d.frame+'  |  Sensor active  |  HR buf: '+d.hr_buf_fill+'/'+d.hr_buf_size;

      // ── Breathing ──────────────────────────────────────────────────────────
      var brCard = document.getElementById('br-card');
      if(d.br_valid){
        document.getElementById('br-val').innerText = d.br_bpm.toFixed(1);
        document.getElementById('br-cm').innerText  = 'CM: '+d.cm_breath.toFixed(3);
        brCard.classList.remove('card-invalid');
      } else {
        document.getElementById('br-val').innerText = '0';
        document.getElementById('br-cm').innerText  = 'no signal';
        brCard.classList.add('card-invalid');
      }

      // ── Heart rate ─────────────────────────────────────────────────────────
      var hrCard = document.getElementById('hr-card');
      if(d.hr_valid){
        document.getElementById('hr-val').innerText = d.hr_bpm.toFixed(1);
        document.getElementById('hr-cm').innerText  = 'CM: '+d.cm_heart.toFixed(3);
        document.getElementById('hr-quality').innerText =
          'raw='+d.hr_raw.toFixed(1)+' med='+d.hr_median.toFixed(1);
        hrCard.classList.remove('card-invalid');
      } else {
        document.getElementById('hr-val').innerText = '0';
        document.getElementById('hr-cm').innerText  = 'no signal';
        document.getElementById('hr-quality').innerText = '';
        hrCard.classList.add('card-invalid');
      }

      document.getElementById('rng-val').innerText = d.rangeBin;
      document.getElementById('energy-val').innerText =
        'E_br:'+d.energy_breath.toFixed(1)+' E_hr:'+d.energy_heart.toFixed(2);

      var motionOn = d.motion > 0.5;
      document.getElementById('motion-val').innerText = motionOn ? 'YES' : 'NO';
      document.getElementById('motion-card').style.background = motionOn ? '#2a0a0a' : '#0f1a0f';

      Plotly.restyle('br-chart',{y:[d.br_wave]},[0]);
      Plotly.restyle('hr-chart',{y:[d.hr_wave]},[0]);
    });
  </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML)


# ─────────────────────────────────────────────────────────────────────────────
# Serial config
# ─────────────────────────────────────────────────────────────────────────────
def read_response(port, timeout=2.0):
    buf = ''
    start = time.time()
    while time.time() - start < timeout:
        if port.in_waiting:
            buf += port.read(port.in_waiting).decode(errors='ignore')
            if 'Done' in buf or 'Error' in buf or 'not recognized' in buf:
                break
        time.sleep(0.05)
    return buf.strip()


def serialConfig(configFileName):
    print(f"[*] Opening CLI  port : {CLI_PORT}")
    print(f"[*] Opening Data port : {DATA_PORT}\n")
    CLIport  = serial.Serial(CLI_PORT,  BAUD_CLI,  timeout=1)
    Dataport = serial.Serial(DATA_PORT, BAUD_DATA, timeout=1)
    time.sleep(3)
    CLIport.reset_input_buffer()
    Dataport.reset_input_buffer()
    for line in open(configFileName):
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        CLIport.write((line + '\r\n').encode())
        resp = read_response(CLIport)
        print(f"  [{'✓' if 'Done' in resp else '✗'}] {line}")
    print("\n[✓] Sensor RUNNING\n")
    return CLIport, Dataport


def safe_float(v, fallback=0.0):
    return fallback if (math.isnan(v) or math.isinf(v)) else v


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────
def readAndParseVitalSigns(Dataport, PKTLEN):
    global byteBuffer, byteBufferLength

    MAGIC   = [2, 1, 4, 3, 6, 5, 8, 7]
    MAX_BUF = 2**15

    out = {
        'breathRate_bpm': 0.0, 'heartRate_bpm': 0.0,
        'cm_breath': 0.0,      'cm_heart': 0.0,
        'energy_breath': 0.0,  'energy_heart': 0.0,
        'rangeBinValue': 0.0,
        'filterBreath': 0.0,   'filterHeart': 0.0,
        'rangeBin': 0,         'motionFlag': 0.0,
        'hasData': False
    }

    avail = Dataport.in_waiting
    raw   = Dataport.read(avail) if avail > 0 else Dataport.read(1)
    vec   = np.frombuffer(raw, dtype='uint8')
    n     = len(vec)
    if n > 0 and byteBufferLength + n < MAX_BUF:
        byteBuffer[byteBufferLength:byteBufferLength + n] = vec
        byteBufferLength += n

    if byteBufferLength < PKTLEN:
        return 0, 0, out

    locs   = np.where(byteBuffer[:byteBufferLength] == MAGIC[0])[0]
    starts = [l for l in locs
              if l + 8 <= byteBufferLength and
              np.all(byteBuffer[l:l + 8] == MAGIC)]

    if not starts:
        keep = min(7, byteBufferLength)
        byteBuffer[:keep] = byteBuffer[byteBufferLength - keep:byteBufferLength]
        byteBufferLength  = keep
        return 0, 0, out

    s = starts[0]
    if s > 0:
        byteBuffer[:byteBufferLength - s] = byteBuffer[s:byteBufferLength]
        byteBuffer[byteBufferLength - s:] = 0
        byteBufferLength -= s

    if byteBufferLength < PKTLEN:
        return 0, 0, out

    pkt = bytes(byteBuffer[:PKTLEN])

    frameNumber = struct.unpack_from('<I', pkt, FRAME_NUM_BYTE_OFFSET)[0]
    rangeBin    = struct.unpack_from('<H', pkt, RANGEBIN_BYTE_OFFSET)[0]

    def f32(offset):
        return struct.unpack_from('<f', pkt, offset)[0] if offset + 4 <= PKTLEN else 0.0

    out['rangeBin']       = int(rangeBin)
    out['rangeBinValue']  = safe_float(f32(B_RANGE_BIN_VALUE))
    out['filterBreath']   = safe_float(f32(B_BREATH_WFM))
    out['filterHeart']    = safe_float(f32(B_HEART_WFM))
    out['heartRate_bpm']  = safe_float(f32(B_HEART_FFT))
    out['breathRate_bpm'] = safe_float(f32(B_BREATH_FFT))
    out['cm_breath']      = safe_float(f32(B_CM_BREATH))
    out['cm_heart']       = safe_float(f32(B_CM_HEART))
    out['energy_breath']  = safe_float(f32(B_ENERGY_BREATH))
    out['energy_heart']   = safe_float(f32(B_ENERGY_HEART))
    out['motionFlag']     = safe_float(f32(B_MOTION_FLAG))
    out['hasData']        = True

    byteBuffer[:byteBufferLength - PKTLEN] = byteBuffer[PKTLEN:byteBufferLength]
    byteBuffer[byteBufferLength - PKTLEN:] = 0
    byteBufferLength -= PKTLEN

    return 1, frameNumber, out


# ─────────────────────────────────────────────────────────────────────────────
# Radar thread
# ─────────────────────────────────────────────────────────────────────────────
def radar_thread(Dataport, PKTLEN):
    global breathing_wave, heartbeat_wave

    # ── Persistent filter state ───────────────────────────────────────────────
    rangeBinValueEMA = 0.0          # slow EMA on signal energy

    # Heart rate circular buffer — only filled with CLEAN readings.
    # Never pushed into when: motion=1, cm_heart low, value out of 40-180 range.
    # Pre-fill with 0 (not 48) so warmup period forces proper accumulation.
    hr_buffer        = deque(maxlen=HR_BUFFER_LEN)
    hr_buffer_filled = 0            # count of real readings pushed so far
    hr_display_ema   = 0.0          # EMA smoothed over the median output
    hr_invalid_count = 0            # consecutive invalid frames counter
    br_invalid_count = 0            # consecutive invalid frames counter

    # Last emitted valid values — held during the holdoff window
    last_hr_bpm = 0.0
    last_br_bpm = 0.0
    hr_warmed   = False             # True once buffer has enough real readings
    br_zero_count = 0 

    print("[*] Vital signs thread started\n")

    while True:
        try:
            dataOK, frameNumber, out = readAndParseVitalSigns(Dataport, PKTLEN)

            if not dataOK:
                time.sleep(0.01)
                continue
            if not out['hasData']:
                continue

            # ── Waveform buffers ──────────────────────────────────────────────
            breathing_wave.append(float(out['filterBreath']))
            heartbeat_wave.append(float(out['filterHeart']))

            # ── Range-bin energy EMA (alpha=0.1) ──────────────────────────────
            rangeBinValueEMA = (RANGE_BIN_EMA_ALPHA * out['rangeBinValue'] +
                                (1.0 - RANGE_BIN_EMA_ALPHA) * rangeBinValueEMA)

            signal_present = rangeBinValueEMA >= THRESH_RANGE_BIN_VALUE
            motion_flag    = out['motionFlag'] > 0.5

            # ── Heart rate filter ─────────────────────────────────────────────
            #
            # Gate conditions for a reading to enter the buffer:
            #   1. Signal present (rangeBinEMA >= 250)
            #   2. No large motion
            #   3. Confidence metric above threshold
            #   4. Energy above threshold
            #   5. Value is physiologically plausible (40–180 BPM)
            #
            # If ALL gates pass → push into buffer and update EMA display.
            # If ANY gate fails → increment invalid counter but DO NOT push 0
            #                     into the buffer (buffer stays clean).
            hr_raw = out['heartRate_bpm']
            hr_gate_pass = (
                signal_present and
                not motion_flag and
                out['cm_heart']     >= THRESH_CM_HEART and
                out['energy_heart'] >= THRESH_ENERGY_HEART and
                HR_MIN_BPM <= hr_raw <= HR_MAX_BPM
            )

            if hr_gate_pass:
                hr_buffer.append(hr_raw)
                hr_buffer_filled = min(hr_buffer_filled + 1, HR_BUFFER_LEN)
                hr_invalid_count = 0

                # Only trust the median once the buffer has enough real data
                hr_warmed = hr_buffer_filled >= HR_WARMUP_FRAMES

                if hr_warmed:
                    hr_median = float(np.median(list(hr_buffer)))
                    # EMA on top of median → prevents single-BPM display jumps
                    if hr_display_ema == 0.0:
                        hr_display_ema = hr_median   # cold start: accept immediately
                    else:
                        hr_display_ema = (HR_DISPLAY_EMA_ALPHA * hr_median +
                                          (1.0 - HR_DISPLAY_EMA_ALPHA) * hr_display_ema)
                    last_hr_bpm = hr_display_ema
            else:
                hr_invalid_count += 1

            # After holdoff expires: clear last value → display goes to 0
            if hr_invalid_count >= HR_HOLDOFF_FRAMES:
                last_hr_bpm      = 0.0
                hr_display_ema   = 0.0
                hr_buffer_filled = 0
                hr_warmed        = False
                hr_buffer.clear()

            hr_valid = hr_warmed and last_hr_bpm > 0
            hr_bpm   = last_hr_bpm if hr_valid else 0.0

            # ── Breathing rate (simpler — firmware estimate is already stable) ─
            br_valid_raw = (signal_present and
                            out['energy_breath'] >= THRESH_ENERGY_BREATH)

            if br_valid_raw:
                br_invalid_count = 0
                last_br_bpm = out['breathRate_bpm']
            else:
                br_invalid_count += 1

            if br_invalid_count >= BR_HOLDOFF_FRAMES:
                last_br_bpm = 0.0

            br_valid = last_br_bpm > 0
            br_bpm   = last_br_bpm if br_valid else 0.0

            if last_br_bpm == 0.0:
                br_zero_count += 1
            else:
                br_zero_count = 0

            if br_zero_count >= BR_ZERO_KILLS_HR_FRAMES:
                last_hr_bpm      = 0.0
                hr_display_ema   = 0.0
                hr_buffer_filled = 0
                hr_warmed        = False
                hr_buffer.clear()
                hr_valid = False
                hr_bpm   = 0.0

            # ── Debug print ───────────────────────────────────────────────────
            hr_median_disp = float(np.median(list(hr_buffer))) if hr_buffer else 0.0
            print(f"Frame #{frameNumber:05d} | "
                  f"BR={'--' if not br_valid else f'{br_bpm:.1f}'} bpm  "
                  f"HR={'--' if not hr_valid else f'{hr_bpm:.1f}'} bpm "
                  f"(raw={hr_raw:.1f} med={hr_median_disp:.1f} buf={hr_buffer_filled})  "
                  f"rbEMA={rangeBinValueEMA:.0f} "
                  f"E_br={out['energy_breath']:.2f} E_hr={out['energy_heart']:.3f} "
                  f"CM_h={out['cm_heart']:.3f}  motion={out['motionFlag']:.0f}")

            socketio.emit('vital_signs', {
                'frame'        : int(frameNumber),
                'br_bpm'       : round(br_bpm, 1),
                'hr_bpm'       : round(hr_bpm, 1),
                'hr_raw'       : round(hr_raw, 1),
                'hr_median'    : round(hr_median_disp, 1),
                'hr_buf_fill'  : hr_buffer_filled,
                'hr_buf_size'  : HR_BUFFER_LEN,
                'br_valid'     : br_valid,
                'hr_valid'     : hr_valid,
                'cm_breath'    : round(out['cm_breath'], 3),
                'cm_heart'     : round(out['cm_heart'],  3),
                'energy_breath': round(out['energy_breath'], 2),
                'energy_heart' : round(out['energy_heart'],  3),
                'rangeBin'     : out['rangeBin'],
                'motion'       : out['motionFlag'],
                'br_wave'      : list(breathing_wave),
                'hr_wave'      : list(heartbeat_wave)
            })

        except Exception as e:
            print(f"[!] Error in radar_thread: {e}")
            time.sleep(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    PKTLEN, _ = parse_cfg_for_pktlen(CONFIG_FILE)
    CLIport, Dataport = serialConfig(CONFIG_FILE)

    t = threading.Thread(target=radar_thread, args=(Dataport, PKTLEN), daemon=True)
    t.start()

    print("[*] Open http://<raspberry-pi-ip>:5000 in your browser\n")
    try:
        socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n[*] Stopping...")
        CLIport.write(b'sensorStop\r\n')
        CLIport.close()
        Dataport.close()
        print("[✓] Done.")

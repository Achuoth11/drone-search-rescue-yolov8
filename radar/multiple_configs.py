import serial
import time
import math
import struct

CLI_PORT  = '/dev/iwr_cli'
DATA_PORT = '/dev/iwr_data'
BAUD_CLI  = 115200
BAUD_DATA = 921600

CONFIG_SCHEDULE = [
    ('/home/iiot1/Radar/20fps.cfg', 60),
    ('/home/iiot1/Radar/30fps.cfg', 60),
    ('/home/iiot1/Radar/10fps.cfg', 60),
]

MAGIC          = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])
SETTLE_TIME    = 3    # seconds to wait after sensorStart before reading
STOP_WAIT      = 2    # seconds to wait after sensorStop before next config
CLI_FLUSH_WAIT = 0.5  # seconds to let CLI drain before flushing

def pow2roundup(x):
    y = 1
    while x > y:
        y *= 2
    return y

def parse_pktlen(cfg_file):
    p = {}
    with open(cfg_file, 'r') as f:
        for line in f:
            tok = line.strip().split()
            if not tok or tok[0].startswith('%'):
                continue
            if tok[0] == 'profileCfg':
                p['numAdcSamples']    = int(tok[10])
                p['digOutSampleRate'] = float(tok[11])
                p['freqSlopeConst']   = float(tok[8])
            elif tok[0] == 'vitalSignsCfg':
                p['rangeStart'] = float(tok[1])
                p['rangeEnd']   = float(tok[2])

    nBins  = pow2roundup(p['numAdcSamples'])
    fsc    = (48.0 * p['freqSlopeConst'] * (2**26) * 1e3) / (3.6e9 * 900.0)
    t_c    = 1e3 * p['numAdcSamples'] / p['digOutSampleRate']
    bw     = fsc * t_c
    rmax   = (t_c * p['digOutSampleRate'] * 3e8) / (2.0 * bw * 1e9)
    rbin   = rmax / nBins
    n_proc = math.floor(p['rangeEnd'] / rbin) - math.floor(p['rangeStart'] / rbin) + 1

    pktlen = 40 + 8 + 4*n_proc + 8 + 128
    if pktlen % 32 != 0:
        pktlen = math.ceil(pktlen / 32) * 32

    print(f"  [cfg] n_bins={n_proc}  PKTLEN={pktlen} bytes")
    return pktlen

def cli_send(cli, cmd, wait=0.2):
    """Send a single CLI command and print the radar's response.
    Returns (response_str, crashed) where crashed=True if firmware exception detected."""
    cli.write((cmd + '\n').encode())
    time.sleep(wait)
    response = b''
    while cli.in_waiting:
        response += cli.read(cli.in_waiting)
        time.sleep(0.05)
    resp_str = response.decode(errors='replace').strip()
    crashed  = 'exception' in resp_str.lower()
    if resp_str:
        print(f"    >> {cmd}")
        for line in resp_str.splitlines():
            print(f"    << {line}")
        if crashed:
            print(f"  [FATAL] Firmware exception detected! Radar needs power cycle.")
    else:
        print(f"    >> {cmd}  (no response)")
    return resp_str, crashed

def stop_sensor(cli, data):
    """Cleanly stop the sensor and flush both ports before next config."""
    print("  [stop] Sending sensorStop...")
    cli_send(cli, 'sensorStop', wait=1.0)
    time.sleep(STOP_WAIT)
    time.sleep(CLI_FLUSH_WAIT)
    cli.reset_input_buffer()
    data.reset_input_buffer()
    print("  [stop] Sensor stopped and buffers flushed.")

def send_cfg(cli, cfg_file):
    """Send config file line by line. Returns False if a firmware crash is detected."""
    print(f"\n{'='*60}")
    print(f"  Loading: {cfg_file}")
    print(f"{'='*60}")
    with open(cfg_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.lower() in ('sensorstart', 'sensorstop'):
                print(f"    -- skipping in-file '{line}' (sent explicitly)")
                continue
            resp, crashed = cli_send(cli, line, wait=0.2)
            if crashed:
                return False

    print("  [cfg] Sending sensorStart...")
    time.sleep(0.5)
    resp, crashed = cli_send(cli, 'sensorStart', wait=0.5)
    if crashed:
        return False
    if 'error' in resp.lower() or 'ignored' in resp.lower():
        print(f"  [WARN] sensorStart may have failed — response: {resp!r}")
    return True

def f32(pkt, byte_offset):
    return struct.unpack_from('<f', pkt, byte_offset)[0]

def parse_and_print(pkt, frame_num, cfg_label):
    breath_bpm = f32(pkt, 48 + (12-1)*4)
    heart_bpm  = f32(pkt, 48 + (8-1)*4)
    cm_breath  = f32(pkt, 48 + (15-1)*4)
    cm_heart   = f32(pkt, 48 + (17-1)*4)
    e_breath   = f32(pkt, 48 + (20-1)*4)
    e_heart    = f32(pkt, 48 + (21-1)*4)
    motion     = f32(pkt, 48 + (22-1)*4)
    br_wfm     = f32(pkt, 48 + (6-1)*4)
    hr_wfm     = f32(pkt, 48 + (7-1)*4)

    print(f"[{cfg_label}] Frame {frame_num:05d} | "
          f"Breath: {breath_bpm:5.1f} bpm  Heart: {heart_bpm:5.1f} bpm | "
          f"CM_b: {cm_breath:+6.3f}  CM_h: {cm_heart:+6.3f} | "
          f"E_b: {e_breath:7.2f}  E_h: {e_heart:7.3f} | "
          f"Motion: {int(motion)} | "
          f"Wfm_br: {br_wfm:+.4f}  Wfm_hr: {hr_wfm:+.4f}")

if __name__ == '__main__':
    cli  = serial.Serial(CLI_PORT,  BAUD_CLI,  timeout=1)
    data = serial.Serial(DATA_PORT, BAUD_DATA, timeout=0.1)
    time.sleep(1)

    # Ensure sensor is stopped before we begin (handles dirty restarts)
    print("[*] Ensuring sensor is stopped before starting...")
    cli_send(cli, 'sensorStop', wait=1.0)
    time.sleep(STOP_WAIT)
    cli.reset_input_buffer()
    data.reset_input_buffer()

    for i, (cfg_file, duration) in enumerate(CONFIG_SCHEDULE):
        cfg_label = cfg_file.split('/')[-1].replace('.cfg', '')

        ok = send_cfg(cli, cfg_file)
        if not ok:
            print(f"\n  [SKIP] {cfg_label} — firmware crashed during config.")
            print(f"  [SKIP] Power cycle the radar and restart the script.")
            # Flush and try to continue in case it self-recovered
            time.sleep(2)
            cli.reset_input_buffer()
            data.reset_input_buffer()
            continue

        pktlen = parse_pktlen(cfg_file)

        data.reset_input_buffer()
        print(f"  [settle] Waiting {SETTLE_TIME}s for sensor to start...")
        time.sleep(SETTLE_TIME)
        data.reset_input_buffer()

        buf        = b''
        frame_num  = 0
        bytes_seen = 0
        magic_hits = 0
        t_end      = time.time() + duration
        print()

        while time.time() < t_end:
            chunk = data.read(data.in_waiting or 64)
            if chunk:
                buf        += chunk
                bytes_seen += len(chunk)

            if bytes_seen > 0 and magic_hits == 0 and bytes_seen % 5000 < 64:
                print(f"  [dbg] bytes_seen={bytes_seen}, no magic yet | "
                      f"buf_tail={buf[-8:].hex()}")

            idx = buf.find(MAGIC)
            if idx == -1:
                buf = buf[-8:]
                continue
            if idx > 0:
                buf = buf[idx:]

            magic_hits += 1
            if magic_hits <= 3:
                print(f"  [dbg] Magic hit #{magic_hits} | "
                      f"buf_len={len(buf)} need={pktlen} | "
                      f"header={buf[:16].hex()}")

            if len(buf) < pktlen:
                continue

            pkt       = buf[:pktlen]
            buf       = buf[pktlen:]
            frame_num += 1
            parse_and_print(pkt, frame_num, cfg_label)

        print(f"\n  [{cfg_label}] Done. {frame_num} frames received.")
        print(f"  [dbg] Total bytes_seen={bytes_seen}, magic_hits={magic_hits}")

        if i < len(CONFIG_SCHEDULE) - 1:
            stop_sensor(cli, data)

    print("\n[*] All configs done. Stopping sensor...")
    cli_send(cli, 'sensorStop', wait=1.0)
    time.sleep(1)
    cli.close()
    data.close()
    print("[*] Done.")
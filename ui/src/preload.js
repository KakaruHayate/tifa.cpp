'use strict';
const { contextBridge, ipcRenderer, webUtils } = require('electron');

// Resolve the on-disk path of a File dragged into the window.  Electron >= 30
// removed File.path, so webUtils is the supported route.
function pathForFile(file) {
  try {
    if (webUtils && typeof webUtils.getPathForFile === 'function') return webUtils.getPathForFile(file);
  } catch { /* fall through */ }
  return (file && (file.path || file.name)) || null;
}

contextBridge.exposeInMainWorld('bridge', {
  // config
  getConfig: () => ipcRenderer.invoke('cfg:get'),
  setConfig: (patch) => ipcRenderer.invoke('cfg:set', patch),
  sysInfo: () => ipcRenderer.invoke('sys:info'),

  // engine CLI
  detectCli: () => ipcRenderer.invoke('cli:detect'),
  pickCli: () => ipcRenderer.invoke('cli:pick'),

  // model
  pickModel: () => ipcRenderer.invoke('model:pick'),
  setModel: (p) => ipcRenderer.invoke('model:set', p),
  modelInfo: (p) => ipcRenderer.invoke('model:info', p),
  inspectModel: (p) => ipcRenderer.invoke('model:inspect', p),

  // inputs
  pickFiles: () => ipcRenderer.invoke('inputs:pick-files'),
  pickFolder: () => ipcRenderer.invoke('inputs:pick-folder'),
  scanPaths: (paths) => ipcRenderer.invoke('inputs:scan', paths),
  pathForFile,

  // phone sources
  pickCsv: () => ipcRenderer.invoke('phones:pick-csv'),
  pickTextgridDir: () => ipcRenderer.invoke('phones:pick-textgrid-dir'),

  // output
  pickOutDir: (current) => ipcRenderer.invoke('output:pick-dir', current),
  ensureDir: (dir) => ipcRenderer.invoke('output:ensure-dir', dir),
  openPath: (target) => ipcRenderer.invoke('shell:open', target),
  saveCsv: (defaultName, text) => ipcRenderer.invoke('csv:save', defaultName, text),

  // reads for the preview pane (main-process guarded)
  readText: (p) => ipcRenderer.invoke('fs:read-text', p),
  readBytes: (p) => ipcRenderer.invoke('fs:read-bytes', p),
  exists: (p) => ipcRenderer.invoke('fs:exists', p),

  // run control
  runAlign: (req) => ipcRenderer.invoke('run:align', req),
  cancelRun: () => ipcRenderer.invoke('run:cancel'),
  resetCancel: () => ipcRenderer.invoke('run:reset-cancel'),
  onRunChunk: (cb) => {
    const listener = (_e, payload) => cb(payload);
    ipcRenderer.on('run:chunk', listener);
    return () => ipcRenderer.removeListener('run:chunk', listener);
  },
});

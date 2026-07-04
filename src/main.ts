import './ui/styles.css';
import { ImageEnhancer } from './api/ImageEnhancer';
import type { TaskProgress, TaskStatus, EnhanceParams } from './api/types';

const enhancer = new ImageEnhancer();

// Доступ к экземпляру для автотестов/бенчмарков. В обычной работе скрыт; включается
// в dev-режиме или явным флагом ?debug в URL (по умолчанию в проде не доступен).
if (import.meta.env.DEV || new URLSearchParams(location.search).has('debug')) {
  (window as unknown as { __enhancer: ImageEnhancer }).__enhancer = enhancer;
}

// --- DOM ---
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const dropZone = $('dropZone');
const fileInput = $<HTMLInputElement>('fileInput');
const browseBtn = $('browseBtn');
const samplesBox = $('samplesBox');

const statusBox = $('statusBox');
const progressFill = $('progressFill');
const statusLabel = $('statusLabel');
const statusPercent = $('statusPercent');
const abortBtn = $('abortBtn');

const resultBox = $('resultBox');
const beforeImg = $<HTMLImageElement>('beforeImg');
const afterImg = $<HTMLImageElement>('afterImg');
const paramsBox = $('paramsBox');
const metaBox = $('metaBox');
const downloadBtn = $('downloadBtn');
const resetBtn = $('resetBtn');

const errorBox = $('errorBox');

// --- Состояние текущей задачи ---
let currentTaskId: string | null = null;
let beforeUrl: string | null = null;
let afterUrl: string | null = null;
let resultBlob: Blob | null = null;
let startTime = 0;

const STATUS_LABELS: Record<TaskStatus, string> = {
  queued: 'В очереди',
  decoding: 'Декодирование',
  analyzing: 'Подбор параметров (ИИ)',
  enhancing: 'Применение коррекции',
  done: 'Готово',
  aborted: 'Прервано',
  error: 'Ошибка',
};

// --- Утилиты ---
function revoke(url: string | null) {
  if (url) URL.revokeObjectURL(url);
}

function show(el: HTMLElement, visible: boolean) {
  el.hidden = !visible;
}

function resetView() {
  show(statusBox, false);
  show(resultBox, false);
  show(errorBox, false);
  show(dropZone, true);
  show(samplesBox, true);
  progressFill.style.width = '0';
  revoke(beforeUrl);
  revoke(afterUrl);
  beforeUrl = afterUrl = null;
  resultBlob = null;
  currentTaskId = null;
}

function renderParams(p: EnhanceParams) {
  const fmt = (x: number) => (x >= 0 ? '+' : '') + x.toFixed(2);
  paramsBox.innerHTML = `
    <div class="param-chip"><div class="label">Яркость</div><div class="value">${fmt(p.brightness)}</div></div>
    <div class="param-chip"><div class="label">Контраст</div><div class="value">×${p.contrast.toFixed(2)}</div></div>
    <div class="param-chip"><div class="label">Цветность</div><div class="value">×${p.saturation.toFixed(2)}</div></div>
  `;
}

// --- Обработка статусов задачи (событие API) ---
enhancer.addEventListener('statuschange', (e) => {
  const { taskId, status, progress, message, params, error } = (
    e as CustomEvent<TaskProgress>
  ).detail;
  if (taskId !== currentTaskId) return;

  progressFill.style.width = `${progress}%`;
  statusLabel.textContent = message ?? STATUS_LABELS[status];
  statusPercent.textContent = `${Math.round(progress)}%`;

  if (params) renderParams(params);

  if (status === 'done') {
    void onDone(taskId, params);
  } else if (status === 'error') {
    showError(error ?? 'Неизвестная ошибка');
  } else if (status === 'aborted') {
    resetView();
  }
});

async function onDone(taskId: string, params?: EnhanceParams) {
  const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
  resultBlob = (await enhancer.getResult(taskId, 'blob')) as Blob;
  afterUrl = URL.createObjectURL(resultBlob);
  afterImg.src = afterUrl;

  if (params) renderParams(params);

  const source = enhancer.getStatus(taskId).source;
  const srcLabel =
    source === 'model' ? '🧠 Параметры подобраны ИИ-моделью' : '⚙️ Базовый авто-алгоритм';

  metaBox.textContent = `${srcLabel} · Время: ${elapsed} с · Результат: ${(
    resultBlob.size / 1024
  ).toFixed(0)} КБ`;

  show(statusBox, false);
  show(resultBox, true);
}

function showError(text: string) {
  errorBox.textContent = `Ошибка: ${text}`;
  show(statusBox, false);
  show(errorBox, true);
  // Возвращаем зону загрузки, чтобы можно было сразу выбрать другой файл
  show(dropZone, true);
  show(samplesBox, true);
}

// --- Запуск обработки ---
async function handleFile(file: File) {
  resetView();
  show(dropZone, false);
  show(samplesBox, false);
  show(statusBox, true);
  errorBox.textContent = '';

  revoke(beforeUrl);
  beforeUrl = URL.createObjectURL(file);
  beforeImg.src = beforeUrl;

  startTime = performance.now();
  try {
    currentTaskId = await enhancer.createTask(file, { outputType: 'image/jpeg', quality: 0.92 });
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err));
  }
}

// --- События UI ---
browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) void handleFile(file);
  fileInput.value = '';
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer?.files?.[0];
  if (file) void handleFile(file);
});

abortBtn.addEventListener('click', () => {
  if (currentTaskId) void enhancer.abortTask(currentTaskId);
});

// Кнопки «Попробовать на примере»
samplesBox.querySelectorAll<HTMLButtonElement>('.sample-card').forEach((btn) => {
  btn.addEventListener('click', async () => {
    const name = btn.dataset.sample!;
    try {
      const resp = await fetch(`samples/${name}`);
      if (!resp.ok) throw new Error(`Не удалось загрузить пример (${resp.status})`);
      const blob = await resp.blob();
      void handleFile(new File([blob], name, { type: 'image/jpeg' }));
    } catch (err) {
      showError(err instanceof Error ? err.message : String(err));
    }
  });
});

downloadBtn.addEventListener('click', () => {
  if (!resultBlob) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(resultBlob);
  a.download = 'enhanced.jpg';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

resetBtn.addEventListener('click', resetView);

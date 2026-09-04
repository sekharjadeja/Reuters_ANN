/* ==========================================================================
   REUTERS FINANCIAL AI - CLIENTSIDE LOGIC & CHART CONTROLLER
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const tabButtons = document.querySelectorAll('.nav-tab');
  const tabPanels = document.querySelectorAll('.tab-panel');

  const articleInput = document.getElementById('article-input');
  const charCounter = document.getElementById('char-counter');
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdVal = document.getElementById('threshold-val');
  const btnClassify = document.getElementById('btn-classify');
  const btnClear = document.getElementById('btn-clear');
  const btnRandomTest = document.getElementById('btn-random-test');

  const sampleChipsContainer = document.getElementById('sample-chips-container');
  const groundTruthBanner = document.getElementById('ground-truth-banner');
  const gtFileId = document.getElementById('gt-file-id');
  const gtTagsContainer = document.getElementById('gt-tags-container');

  const resultsPlaceholder = document.getElementById('results-placeholder');
  const resultsContent = document.getElementById('results-content');
  const activeTopicsContainer = document.getElementById('active-topics-container');
  const probabilityMetersContainer = document.getElementById('probability-meters-container');
  const tokensContainer = document.getElementById('tokens-container');
  const predictedCountBadge = document.getElementById('predicted-count-badge');
  const latencyVal = document.getElementById('latency-val');

  const categorySearch = document.getElementById('category-search');
  const categoryGrid = document.getElementById('category-grid');

  // State
  let lastPredictionData = null;
  let allCategories = [];
  let accuracyChartInstance = null;
  let lossChartInstance = null;

  // ==========================================================================
  // TAB NAVIGATION
  // ==========================================================================
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.dataset.tab;

      tabButtons.forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      tabPanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const targetPanel = document.getElementById(`panel-${targetTab}`);
      if (targetPanel) targetPanel.classList.add('active');

      if (targetTab === 'analytics') {
        renderAnalyticsCharts();
      }
    });
  });

  // ==========================================================================
  // INPUT COUNTERS & RESETS
  // ==========================================================================
  function updateInputStats() {
    const text = articleInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    const chars = articleInput.value.length;
    charCounter.textContent = `${words} words | ${chars} characters`;
  }

  articleInput.addEventListener('input', updateInputStats);

  btnClear.addEventListener('click', () => {
    articleInput.value = '';
    updateInputStats();
    groundTruthBanner.style.display = 'none';
    resultsContent.style.display = 'none';
    resultsPlaceholder.style.display = 'flex';
    latencyVal.textContent = '-- ms';
    document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
    lastPredictionData = null;
  });

  // ==========================================================================
  // SAMPLES PRESETS
  // ==========================================================================
  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      const samples = await res.json();

      sampleChipsContainer.innerHTML = '';
      samples.forEach((sample, i) => {
        const chip = document.createElement('button');
        chip.className = 'sample-chip';
        chip.innerHTML = `<span>📌</span> ${sample.title}`;
        chip.title = sample.description;
        chip.addEventListener('click', () => {
          document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));
          chip.classList.add('active');
          articleInput.value = sample.text;
          updateInputStats();
          groundTruthBanner.style.display = 'none';
          triggerClassification();
        });
        sampleChipsContainer.appendChild(chip);
      });
    } catch (err) {
      console.error('Failed to load samples:', err);
    }
  }

  // ==========================================================================
  // RANDOM TEST DOCUMENT WITH GROUND TRUTH
  // ==========================================================================
  btnRandomTest.addEventListener('click', async () => {
    btnRandomTest.disabled = true;
    btnRandomTest.innerHTML = `<span class="btn-spinner" style="display:inline-block"></span> Loading...`;

    try {
      const res = await fetch('/api/random_test');
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      articleInput.value = data.text;
      updateInputStats();
      document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('active'));

      // Ground truth banner
      gtFileId.textContent = data.file_id;
      gtTagsContainer.innerHTML = '';
      if (data.ground_truth && data.ground_truth.length > 0) {
        data.ground_truth.forEach(cat => {
          const tag = document.createElement('span');
          tag.className = 'gt-tag';
          tag.textContent = `#${cat}`;
          gtTagsContainer.appendChild(tag);
        });
      } else {
        gtTagsContainer.innerHTML = '<span class="text-muted" style="font-size:0.75rem;">(No ground truth topics assigned in cats.txt)</span>';
      }
      groundTruthBanner.style.display = 'block';

      triggerClassification();
    } catch (err) {
      alert('Could not fetch test document: ' + err.message);
    } finally {
      btnRandomTest.disabled = false;
      btnRandomTest.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 21h5v-5"/></svg>
        Random Test File
      `;
    }
  });

  // ==========================================================================
  // CLASSIFICATION ENGINE
  // ==========================================================================
  btnClassify.addEventListener('click', () => triggerClassification());

  async function triggerClassification() {
    const text = articleInput.value.trim();
    if (!text) {
      alert('Please enter or select a financial news story to classify.');
      articleInput.focus();
      return;
    }

    const currentThreshold = parseFloat(thresholdSlider.value);
    btnClassify.classList.add('loading');

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, threshold: currentThreshold })
      });

      const data = await res.json();
      if (data.error) throw new Error(data.error);

      lastPredictionData = data;
      renderPredictions(data, currentThreshold);
    } catch (err) {
      alert('Prediction failed: ' + err.message);
    } finally {
      btnClassify.classList.remove('loading');
    }
  }

  // ==========================================================================
  // DYNAMIC THRESHOLD SLIDER
  // ==========================================================================
  thresholdSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    thresholdVal.textContent = val.toFixed(2);

    // If we have previous prediction data, update display dynamically in real-time
    if (lastPredictionData && lastPredictionData.top_candidates) {
      renderPredictions(lastPredictionData, val);
    }
  });

  function renderPredictions(data, threshold) {
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = 'block';

    latencyVal.textContent = `${data.latency_ms} ms`;

    // Filter candidates by current threshold
    const active = data.top_candidates.filter(c => c.probability >= threshold);
    predictedCountBadge.textContent = `${active.length} Active Topic${active.length === 1 ? '' : 's'}`;

    // Active Topic Badges
    activeTopicsContainer.innerHTML = '';
    if (active.length > 0) {
      active.forEach(item => {
        const badge = document.createElement('div');
        badge.className = 'topic-badge';
        badge.innerHTML = `
          <span>🏷️ ${item.category}</span>
          <span class="topic-prob">${(item.probability * 100).toFixed(1)}%</span>
        `;
        activeTopicsContainer.appendChild(badge);
      });
    } else {
      activeTopicsContainer.innerHTML = `
        <div class="no-topics-warning">
          ⚠️ No topics meet the current confidence threshold (${threshold.toFixed(2)}). Lower the threshold slider to view marginal candidates.
        </div>
      `;
    }

    // Probability meters for top candidates
    probabilityMetersContainer.innerHTML = '';
    data.top_candidates.forEach(cand => {
      const isSelected = cand.probability >= threshold;
      const row = document.createElement('div');
      row.className = `prob-meter-row ${isSelected ? 'selected' : ''}`;

      const pct = (cand.probability * 100).toFixed(1);
      const isHigh = cand.probability >= 0.7;

      row.innerHTML = `
        <div class="prob-meta">
          <span class="prob-name">
            ${isSelected ? '<span class="check-icon">✓</span>' : ''}
            ${cand.category}
          </span>
          <span class="prob-val ${isSelected ? 'text-cyan' : 'text-muted'}">${pct}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill ${isHigh ? 'high' : ''}" style="width: ${Math.max(2, pct)}%"></div>
        </div>
      `;
      probabilityMetersContainer.appendChild(row);
    });

    // Salient Tokens
    tokensContainer.innerHTML = '';
    if (data.tokens && data.tokens.length > 0) {
      data.tokens.forEach(tok => {
        const chip = document.createElement('span');
        chip.className = 'token-chip';
        chip.textContent = tok;
        tokensContainer.appendChild(chip);
      });
    } else {
      tokensContainer.innerHTML = '<span class="text-muted" style="font-size:0.75rem;">No high-frequency financial tokens detected</span>';
    }
  }

  // ==========================================================================
  // TAB 2: MODEL ANALYTICS CHARTS (Chart.js)
  // ==========================================================================
  async function renderAnalyticsCharts() {
    if (accuracyChartInstance && lossChartInstance) return; // already rendered

    try {
      const res = await fetch('/api/metrics');
      const data = await res.json();
      const history = data.history || {};

      const epochs = Array.from({ length: 10 }, (_, i) => `Epoch ${i + 1}`);

      // Chart defaults for modern dark UI
      Chart.defaults.color = '#94a3b8';
      Chart.defaults.font.family = "'Inter', sans-serif";

      // 1. Accuracy Chart
      const accCtx = document.getElementById('accuracyChart').getContext('2d');
      accuracyChartInstance = new Chart(accCtx, {
        type: 'line',
        data: {
          labels: epochs,
          datasets: [
            {
              label: 'Train Accuracy',
              data: history.accuracy || [],
              borderColor: '#06b6d4',
              backgroundColor: 'rgba(6, 182, 212, 0.1)',
              borderWidth: 2.5,
              tension: 0.35,
              pointBackgroundColor: '#38bdf8',
              fill: true
            },
            {
              label: 'Val Accuracy',
              data: history.val_accuracy || [],
              borderColor: '#6366f1',
              backgroundColor: 'rgba(99, 102, 241, 0.05)',
              borderWidth: 2.5,
              borderDash: [5, 5],
              tension: 0.35,
              pointBackgroundColor: '#818cf8',
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top', labels: { boxWidth: 12, usePointStyle: true } }
          },
          scales: {
            y: {
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { callback: v => (v * 100).toFixed(0) + '%' }
            },
            x: { grid: { display: false } }
          }
        }
      });

      // 2. Loss Chart
      const lossCtx = document.getElementById('lossChart').getContext('2d');
      lossChartInstance = new Chart(lossCtx, {
        type: 'line',
        data: {
          labels: epochs,
          datasets: [
            {
              label: 'Train Loss',
              data: history.loss || [],
              borderColor: '#10b981',
              backgroundColor: 'rgba(16, 185, 129, 0.1)',
              borderWidth: 2.5,
              tension: 0.35,
              pointBackgroundColor: '#34d399',
              fill: true
            },
            {
              label: 'Val Loss',
              data: history.val_loss || [],
              borderColor: '#f59e0b',
              backgroundColor: 'rgba(245, 158, 11, 0.05)',
              borderWidth: 2.5,
              borderDash: [5, 5],
              tension: 0.35,
              pointBackgroundColor: '#fbbf24',
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'top', labels: { boxWidth: 12, usePointStyle: true } }
          },
          scales: {
            y: {
              grid: { color: 'rgba(255, 255, 255, 0.05)' }
            },
            x: { grid: { display: false } }
          }
        }
      });
    } catch (err) {
      console.error('Failed to load metrics or render charts:', err);
    }
  }

  // ==========================================================================
  // TAB 3: CATEGORY EXPLORER
  // ==========================================================================
  async function loadCategories() {
    try {
      const res = await fetch('/api/categories');
      const data = await res.json();
      allCategories = data.categories || [];
      renderCategories(allCategories);
    } catch (err) {
      console.error('Failed to load categories:', err);
    }
  }

  function renderCategories(cats) {
    categoryGrid.innerHTML = '';
    if (cats.length === 0) {
      categoryGrid.innerHTML = '<div class="text-muted">No matching topics found</div>';
      return;
    }
    cats.forEach(c => {
      const item = document.createElement('div');
      item.className = 'cat-item';
      item.innerHTML = `
        <span class="cat-name">${c.name}</span>
        <span class="cat-count">${c.count} docs</span>
      `;
      item.addEventListener('click', () => {
        // Quick filter search
        categorySearch.value = c.name;
        filterCategories();
      });
      categoryGrid.appendChild(item);
    });
  }

  categorySearch.addEventListener('input', () => filterCategories());

  function filterCategories() {
    const q = categorySearch.value.trim().toLowerCase();
    const filtered = allCategories.filter(c => c.name.toLowerCase().includes(q));
    renderCategories(filtered);
  }

  // ==========================================================================
  // INITIALIZATION
  // ==========================================================================
  loadSamples();
  loadCategories();
  updateInputStats();
});

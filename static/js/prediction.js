/**
 * GloHealth AI · Symptom Checker & Clinical Telemetry (Stitch)
 */

document.addEventListener('DOMContentLoaded', function() {
    // 1. DOM References
    const symptomForm = document.getElementById('symptomForm');
    const symptomCards = document.querySelectorAll('.gh-symptom');
    const selectedCount = document.getElementById('selectedCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const clearAll = document.getElementById('clearAll');
    const symptomSearch = document.getElementById('symptomSearch');
    const categoryChips = document.getElementById('categoryChips');
    const resultsModal = document.getElementById('resultsModal');
    const modalBody = document.getElementById('modalBody');
    const closeModal = document.getElementById('closeModal');
    const userDivision = document.getElementById('userDivision');

    let userLocation = { lat: null, long: null };
    // Cache last result so we can re-render it when language changes
    let lastPredictions = null;
    let lastLocation = {};

    // i18n helper: get translated string, fallback to key
    function t(key) {
        var lang = (window.GH_LANG || 'en');
        var dict = (window.GH_I18N && window.GH_I18N[lang]) || {};
        return dict[key] !== undefined ? dict[key] : key;
    }

    // Request geolocation in background if permitted
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            pos => {
                userLocation.lat = pos.coords.latitude;
                userLocation.long = pos.coords.longitude;
            },
            err => { /* Soft ignore permission denial */ },
            { timeout: 5000 }
        );
    }

    // 2. Checkbox & Card Toggle Handling
    symptomCards.forEach(card => {
        card.addEventListener('click', function(e) {
            if (e.target.closest('.gh-symptom__slider')) {
                return;
            }
            const checkbox = this.querySelector('.symptom-check');
            if (checkbox) {
                checkbox.checked = !checkbox.checked;
                this.classList.toggle('is-checked', checkbox.checked);
                updateCount();
            }
        });
    });

    // 3. Slider Handling
    document.querySelectorAll('.pain-slider').forEach(slider => {
        slider.addEventListener('input', function(e) {
            const valDisplay = this.nextElementSibling;
            if (valDisplay && valDisplay.classList.contains('gh-symptom__slider-value')) {
                valDisplay.textContent = this.value;
            }
        });
        slider.addEventListener('click', function(e) {
            e.stopPropagation();
        });
    });

    // 4. Search Filter
    if (symptomSearch) {
        symptomSearch.addEventListener('input', function() {
            const query = this.value.trim().toLowerCase();
            symptomCards.forEach(card => {
                const label = card.getAttribute('data-label') || '';
                const key = card.getAttribute('data-key') || '';
                const matches = label.includes(query) || key.includes(query);
                card.style.display = matches ? '' : 'none';
            });

            // Hide/show category group containers if empty
            document.querySelectorAll('[data-category-group]').forEach(group => {
                const visibleCards = group.querySelectorAll('.gh-symptom:not([style*="display: none"])');
                group.style.display = visibleCards.length > 0 ? '' : 'none';
            });
        });
    }

    // 5. Category Chips Filter
    if (categoryChips) {
        categoryChips.addEventListener('click', function(e) {
            const chip = e.target.closest('.gh-chip');
            if (!chip) return;

            categoryChips.querySelectorAll('.gh-chip').forEach(c => c.classList.remove('is-active'));
            chip.classList.add('is-active');

            const selectedCat = chip.getAttribute('data-category');
            document.querySelectorAll('[data-category-group]').forEach(group => {
                const catName = group.getAttribute('data-category-group');
                if (selectedCat === 'all' || catName === selectedCat) {
                    group.style.display = '';
                } else {
                    group.style.display = 'none';
                }
            });
        });
    }

    // 6. Clear Handler
    function clearAllSymptoms() {
        symptomCards.forEach(card => {
            const checkbox = card.querySelector('.symptom-check');
            if (checkbox) checkbox.checked = false;
            card.classList.remove('is-checked');
            const slider = card.querySelector('.pain-slider');
            if (slider) slider.value = 5;
            const valDisplay = card.querySelector('.gh-symptom__slider-value');
            if (valDisplay) valDisplay.textContent = '5';
        });
        updateCount();
    }

    if (clearBtn) clearBtn.addEventListener('click', clearAllSymptoms);
    if (clearAll) clearAll.addEventListener('click', clearAllSymptoms);

    // 7. Update Count & Button State
    function updateCount() {
        const checked = document.querySelectorAll('.symptom-check:checked');
        const count = checked.length;
        if (selectedCount) selectedCount.textContent = count;
        if (analyzeBtn) {
            analyzeBtn.disabled = (count === 0);
        }
    }

    // 8. Analyze Trigger
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async function() {
            const checkedInputs = Array.from(document.querySelectorAll('.symptom-check:checked'));
            const symptoms = checkedInputs.map(input => input.value);
            const division = (userDivision && userDivision.value) ? userDivision.value : 'Dhaka';

            if (!symptoms.length) return;

            this.classList.add('gh-btn--loading');
            this.disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        symptoms: symptoms,
                        division: division,
                        lat: userLocation.lat,
                        long: userLocation.long
                    })
                });

                const data = await response.json();
                if (data.success && data.predictions && data.predictions.length) {
                    lastPredictions = data.predictions;
                    lastLocation = data.location_factors || {};
                    renderResult(lastPredictions, lastLocation);
                    if (resultsModal) resultsModal.classList.add('is-open');
                } else {
                    throw new Error(data.error || 'Diagnostic evaluation failed.');
                }
            } catch (err) {
                alert('Analysis Error: ' + err.message);
            } finally {
                this.classList.remove('gh-btn--loading');
                updateCount();
            }
        });
    }

    // 9. Modal Close Handlers
    function hideModal() {
        if (resultsModal) resultsModal.classList.remove('is-open');
    }

    if (closeModal) closeModal.addEventListener('click', hideModal);

    if (resultsModal) {
        resultsModal.addEventListener('click', function(e) {
            if (e.target === resultsModal) hideModal();
        });
    }

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && resultsModal && resultsModal.classList.contains('is-open')) {
            hideModal();
        }
    });

    // Re-render modal when language is toggled mid-session
    document.addEventListener('gh:langchange', function() {
        if (lastPredictions && resultsModal && resultsModal.classList.contains('is-open')) {
            renderResult(lastPredictions, lastLocation);
        }
    });

    // 10. Render Assessment Result (i18n-aware)
    function renderResult(predictions, location) {
        if (!modalBody || !predictions || !predictions.length) return;

        const top = predictions[0];
        const pct = Math.round((top.confidence || 0) * 100);
        const divName = location.division || (userDivision ? userDivision.value : 'National Average');

        let differentialHtml = '';
        if (predictions.length > 1) {
            differentialHtml = `
                <div class="gh-modal__section u-mt-3">
                    <div class="gh-modal__section-title">${t('result.differential')}</div>
                    <div class="gh-stack" style="gap: 6px;">
                        ${predictions.slice(1, 3).map(p => {
                            const pPct = Math.round((p.confidence || 0) * 100);
                            return `
                                <div class="gh-row gh-row--between" style="padding: 6px 10px; background: var(--gh-surface-2); border-radius: var(--gh-radius-sm); font-size: 13px;">
                                    <span>${p.disease}</span>
                                    <span class="u-mono" style="font-weight: 600; color: var(--gh-text-muted);">${pPct}%</span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        }

        modalBody.innerHTML = `
            <div class="gh-modal__result">
                <div class="gh-row gh-row--between">
                    <div class="gh-badge gh-badge--alert">${t('result.primary_match')}</div>
                    <span class="u-mono u-muted" style="font-size: 11px;">NODE DHK</span>
                </div>
                <div class="gh-modal__condition u-mt-3">${top.disease}</div>
                <div class="gh-row" style="gap: 12px; margin-top: 8px;">
                    <div class="gh-confidence" style="max-width: 100%;">
                        <div class="gh-confidence__track" style="height: 6px;">
                            <div class="gh-confidence__fill" style="width: ${pct}%;"></div>
                        </div>
                        <span class="gh-confidence__value" style="font-size: 15px; font-weight: 700;">${pct}%</span>
                    </div>
                </div>
            </div>

            ${differentialHtml}

            <div class="gh-modal__section u-mt-3">
                <div class="gh-modal__section-title">${t('result.regional')}</div>
                <p class="u-muted" style="font-size: 13px;">
                    ${t('result.regional_text')} <strong>${divName}</strong>${t('result.intervention')}
                </p>
            </div>

            <div class="gh-modal__section u-mt-3">
                <div class="gh-modal__section-title">${t('result.steps')}</div>
                <ul class="gh-modal__precautions">
                    <li>
                        <span class="material-symbols-outlined">check_circle</span>
                        <span>${t('result.step1')}</span>
                    </li>
                    <li>
                        <span class="material-symbols-outlined">check_circle</span>
                        <span>${t('result.step2')}</span>
                    </li>
                    <li>
                        <span class="material-symbols-outlined">check_circle</span>
                        <span>${t('result.step3')}</span>
                    </li>
                    <li>
                        <span class="material-symbols-outlined">check_circle</span>
                        <span>${t('result.step4')}</span>
                    </li>
                </ul>
            </div>

            <div class="gh-row gh-row--between u-mt-4" style="padding-top: 12px; border-top: 1px solid var(--gh-border-soft);">
                <button type="button" class="gh-btn gh-btn--ghost" id="modalCloseBtn">${t('common.close')}</button>
                <a href="/doctors" class="gh-btn gh-btn--primary">
                    <span class="material-symbols-outlined">person_add</span>
                    ${t('common.book_specialist')}
                </a>
            </div>

            <div class="gh-modal__disclaimer">
                ${t('result.disclaimer')}
            </div>
        `;

        const modalCloseBtn = modalBody.querySelector('#modalCloseBtn');
        if (modalCloseBtn) {
            modalCloseBtn.addEventListener('click', hideModal);
        }
    }
});



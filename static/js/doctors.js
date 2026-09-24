/**
 * GloHealth AI - Doctors Directory Filter Controller
 * Handles client-side live filtering by name/hospital, specialty, and division.
 */
document.addEventListener('DOMContentLoaded', function() {
    var searchInput = document.getElementById('doctorSearch');
    var specialtySelect = document.getElementById('specialtyFilter');
    var divisionSelect = document.getElementById('divisionFilter');
    var clearBtn = document.getElementById('clearFiltersBtn');
    var doctorCards = document.querySelectorAll('.doctor-card');
    var emptyState = document.getElementById('doctorsEmptyState');
    var grid = document.getElementById('doctorsGrid');

    function normalizeTerm(str) {
        return (str || '').toLowerCase().replace(/[^a-z0-9]/g, '');
    }

    function matchSpecialty(docSpec, filterSpec) {
        if (!filterSpec) return true;
        var d = normalizeTerm(docSpec);
        var f = normalizeTerm(filterSpec);
        if (!d || !f) return false;
        if (d === f || d.includes(f) || f.includes(d)) return true;
        var stemLen = Math.min(6, d.length, f.length);
        if (stemLen >= 4 && d.slice(0, stemLen) === f.slice(0, stemLen)) return true;
        return false;
    }

    function matchDivision(docDiv, filterDiv) {
        if (!filterDiv) return true;
        var d = normalizeTerm(docDiv);
        var f = normalizeTerm(filterDiv);
        if (!d || !f) return false;
        if (d === f) return true;
        if ((d === 'chittagong' && f === 'chattogram') || (d === 'chattogram' && f === 'chittagong')) return true;
        if ((d === 'barisal' && f === 'barishal') || (d === 'barishal' && f === 'barisal')) return true;
        return d.includes(f) || f.includes(d);
    }

    function applyFilters() {
        var query = (searchInput ? searchInput.value : '').toLowerCase().trim();
        var selectedSpecialty = (specialtySelect ? specialtySelect.value : '');
        var selectedDivision = (divisionSelect ? divisionSelect.value : '');

        var visibleCount = 0;

        doctorCards.forEach(function(card) {
            var name = (card.getAttribute('data-name') || '').toLowerCase();
            var hospital = (card.getAttribute('data-hospital') || '').toLowerCase();
            var specialty = (card.getAttribute('data-specialty') || '').toLowerCase();
            var division = (card.getAttribute('data-division') || '').toLowerCase();

            var matchesQuery = !query || name.includes(query) || hospital.includes(query) || specialty.includes(query);
            var matchesSpecialtyVal = matchSpecialty(specialty, selectedSpecialty);
            var matchesDivisionVal = matchDivision(division, selectedDivision);

            if (matchesQuery && matchesSpecialtyVal && matchesDivisionVal) {
                card.style.display = '';
                visibleCount++;
            } else {
                card.style.display = 'none';
            }
        });

        if (emptyState) {
            emptyState.style.display = (visibleCount === 0) ? 'block' : 'none';
        }
        if (grid) {
            grid.style.display = (visibleCount === 0) ? 'none' : 'grid';
        }
    }

    if (searchInput) searchInput.addEventListener('input', applyFilters);
    if (specialtySelect) specialtySelect.addEventListener('change', applyFilters);
    if (divisionSelect) divisionSelect.addEventListener('change', applyFilters);

    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            if (searchInput) searchInput.value = '';
            if (specialtySelect) specialtySelect.value = '';
            if (divisionSelect) divisionSelect.value = '';
            applyFilters();
        });
    }
});

function swapDoctorLabels() {
    var lang = localStorage.getItem('glohealth_lang') || 'en';
    document.querySelectorAll('[data-label-en]').forEach(function(el) {
        var val = el.getAttribute(lang === 'bn' ? 'data-label-bn' : 'data-label-en');
        if (val) el.textContent = val;
    });
}
swapDoctorLabels();
document.addEventListener('gh:langchange', swapDoctorLabels);


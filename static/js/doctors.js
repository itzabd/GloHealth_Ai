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

    function applyFilters() {
        var query = (searchInput ? searchInput.value : '').toLowerCase().trim();
        var selectedSpecialty = (specialtySelect ? specialtySelect.value : '').toLowerCase();
        var selectedDivision = (divisionSelect ? divisionSelect.value : '').toLowerCase();

        var visibleCount = 0;

        doctorCards.forEach(function(card) {
            var name = (card.getAttribute('data-name') || '').toLowerCase();
            var hospital = (card.getAttribute('data-hospital') || '').toLowerCase();
            var specialty = (card.getAttribute('data-specialty') || '').toLowerCase();
            var division = (card.getAttribute('data-division') || '').toLowerCase();

            var matchesQuery = !query || name.includes(query) || hospital.includes(query) || specialty.includes(query);
            var matchesSpecialty = !selectedSpecialty || specialty === selectedSpecialty;
            var matchesDivision = !selectedDivision || division === selectedDivision;

            if (matchesQuery && matchesSpecialty && matchesDivision) {
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

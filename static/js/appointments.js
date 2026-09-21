/**
 * GloHealth AI - Appointments Tab Controller
 * Handles tab switching between Upcoming, Past, and Cancelled appointments.
 */
document.addEventListener('DOMContentLoaded', function() {
    var tabChips = document.querySelectorAll('.gh-chip[data-tab]');
    var panels = document.querySelectorAll('[id^="panel-"]');
    var backBtn = document.getElementById('btnBackToUpcoming');

    function switchTab(targetTab) {
        tabChips.forEach(function(chip) {
            if (chip.getAttribute('data-tab') === targetTab) {
                chip.classList.add('is-active');
            } else {
                chip.classList.remove('is-active');
            }
        });

        panels.forEach(function(panel) {
            if (panel.id === 'panel-' + targetTab) {
                panel.style.display = 'block';
                panel.removeAttribute('hidden');
            } else {
                panel.style.display = 'none';
                panel.setAttribute('hidden', 'true');
            }
        });
    }

    tabChips.forEach(function(chip) {
        chip.addEventListener('click', function() {
            var targetTab = this.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });

    if (backBtn) {
        backBtn.addEventListener('click', function() {
            switchTab('upcoming');
        });
    }
});

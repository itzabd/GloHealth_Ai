document.addEventListener('DOMContentLoaded', function() {
    let userLocation = { lat: null, long: null };

    // Symptom card toggle
    document.querySelectorAll('.symptom-card').forEach(card => {
        const checkbox = card.querySelector('.symptom-check');
        card.addEventListener('click', function() {
            checkbox.checked = !checkbox.checked;
            this.classList.toggle('selected', checkbox.checked);
        });
    });

    // Use browser location
    const useLocationBtn = document.getElementById('useLocationBtn');
    const locationStatus = document.getElementById('locationStatus');

    useLocationBtn.addEventListener('click', function() {
        this.disabled = true;
        this.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Detecting...';
        locationStatus.textContent = '';
        locationStatus.classList.remove('text-danger');

        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                position => {
                    userLocation = {
                        lat: position.coords.latitude,
                        long: position.coords.longitude
                    };

                    this.innerHTML = '<i class="bi bi-check-circle"></i> Location Found';
                    this.classList.add('btn-success');
                    this.classList.remove('btn-outline-primary');
                    locationStatus.textContent = 'Your location was successfully detected.';

                    // Simple example mapping (expand as needed)
                    if (userLocation.lat > 23.6 && userLocation.lat < 23.9) {
                        document.getElementById('division').value = 'Dhaka';
                    }
                },
                error => {
                    resetLocationBtn();
                    locationStatus.textContent = 'Error: ' + getGeoError(error.code);
                    locationStatus.classList.add('text-danger');
                },
                { timeout: 10000 }
            );
        } else {
            resetLocationBtn();
            locationStatus.textContent = 'Your browser does not support location services.';
            locationStatus.classList.add('text-danger');
        }
    });

    function resetLocationBtn() {
        useLocationBtn.disabled = false;
        useLocationBtn.innerHTML = '<i class="bi bi-geo-alt"></i> Use My Location';
    }

    // Prediction submission
    document.getElementById('predictBtn').addEventListener('click', async function() {
        const symptoms = Array.from(document.querySelectorAll('.symptom-check:checked'))
            .map(el => el.value);

        if (symptoms.length === 0) {
            alert('Please select at least one symptom.');
            return;
        }

        const division = document.getElementById('division').value;
        if (!division) {
            alert('Please select your division.');
            return;
        }

        const locationData = {
            division: division,
            lat: userLocation.lat,
            long: userLocation.long
        };

        this.disabled = true;
        this.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Processing...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symptoms: symptoms,
                    ...locationData
                })
            });

            const data = await response.json();
            if (data.success) {
                showResultsModal(data.predictions);
            } else {
                throw new Error(data.error || 'Unknown error occurred');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            this.disabled = false;
            this.innerHTML = '<i class="bi bi-heart-pulse"></i> Check Symptoms';
        }
    });

    function getGeoError(code) {
        const errors = { 1: 'Permission denied', 2: 'Position unavailable', 3: 'Timeout' };
        return errors[code] || 'Unknown error';
    }

    function showResultsModal(predictions) {
        const modalHtml = `
            <div class="modal fade" id="resultsModal" tabindex="-1" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header bg-primary text-white">
                            <h5 class="modal-title">Prediction Results</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <h5>Possible Diseases:</h5>
                            <ul class="list-group">
                                ${predictions.map(pred => `
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        <span>${pred.disease}</span>
                                        <span class="badge bg-primary rounded-pill">${pred.probability}</span>
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modal = new bootstrap.Modal(document.getElementById('resultsModal'));
        modal.show();
        document.getElementById('resultsModal').addEventListener('hidden.bs.modal', function() {
            this.remove();
        });
    }
});

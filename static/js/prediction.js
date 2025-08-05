document.addEventListener('DOMContentLoaded', function() {
    // Store location data when obtained
    let userLocation = { lat: null, long: null };

    // Symptom selection
    const symptomCards = document.querySelectorAll('.symptom-card');
    symptomCards.forEach(card => {
        const checkbox = card.querySelector('input');
        card.classList.toggle('selected', checkbox.checked);

        card.addEventListener('click', function(e) {
            if (e.target.tagName !== 'INPUT') {
                checkbox.checked = !checkbox.checked;
                this.classList.toggle('selected', checkbox.checked);
            }
        });
    });

    // Geolocation
    const useLocationBtn = document.getElementById('useLocationBtn');
    const locationStatus = document.getElementById('locationStatus');

    useLocationBtn.addEventListener('click', function() {
        this.disabled = true;
        this.innerHTML = '<span class="spinner-border spinner-border-sm"></span> অবস্থান সনাক্ত করা হচ্ছে...';
        locationStatus.textContent = '';
        locationStatus.classList.remove('text-danger');

        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                position => {
                    userLocation = {
                        lat: position.coords.latitude,
                        long: position.coords.longitude
                    };

                    this.innerHTML = '<i class="bi bi-check-circle"></i> অবস্থান পাওয়া গেছে';
                    this.classList.add('btn-success');
                    this.classList.remove('btn-outline-primary');
                    locationStatus.textContent = 'আপনার অবস্থান সফলভাবে সনাক্ত করা হয়েছে';

                    // Simple location to division mapping
                    if (userLocation.lat > 23.7 && userLocation.lat < 23.9) {
                        document.getElementById('division').value = 'Dhaka';
                    }
                },
                error => {
                    this.disabled = false;
                    this.innerHTML = '<i class="bi bi-geo-alt"></i> আমার অবস্থান ব্যবহার করুন';
                    locationStatus.textContent = 'ত্রুটি: ' + getGeoError(error.code);
                    locationStatus.classList.add('text-danger');
                },
                { timeout: 10000 }
            );
        } else {
            this.disabled = false;
            locationStatus.textContent = 'আপনার ব্রাউজার অবস্থান সেবা সমর্থন করে না';
            locationStatus.classList.add('text-danger');
        }
    });

    // Prediction submission
    const predictBtn = document.getElementById('predictBtn');
    predictBtn.addEventListener('click', async function() {
        const symptoms = Array.from(document.querySelectorAll('input[name="symptoms"]:checked'))
            .map(el => el.value);

        if (symptoms.length === 0) {
            alert('দয়া করে কমপক্ষে একটি লক্ষণ নির্বাচন করুন');
            return;
        }

        const division = document.getElementById('division').value;
        if (!division) {
            alert('দয়া করে আপনার বিভাগ নির্বাচন করুন');
            return;
        }

        const locationData = {
            division: division,
            zipcode: document.getElementById('zipcode').value,
            lat: userLocation.lat,
            long: userLocation.long
        };

        this.disabled = true;
        this.innerHTML = '<span class="spinner-border spinner-border-sm"></span> প্রক্রিয়াকরণ...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': '{{ csrf_token() }}'  // Add CSRF protection
                },
                body: JSON.stringify({
                    symptoms: symptoms,
                    ...locationData
                })
            });

            const data = await response.json();
            if (data.success) {
                // Show results in a modal or redirect
                showResultsModal(data.predictions);
            } else {
                throw new Error(data.error || 'অজানা ত্রুটি ঘটেছে');
            }
        } catch (error) {
            alert('ত্রুটি: ' + error.message);
        } finally {
            this.disabled = false;
            this.innerHTML = '<i class="bi bi-heart-pulse"></i> লক্ষণ পরীক্ষা করুন';
        }
    });

    function getGeoError(code) {
        const errors = {
            1: 'অনুমতি দেওয়া হয়নি',
            2: 'অবস্থান পাওয়া যায়নি',
            3: 'সময়সীমা অতিক্রান্ত'
        };
        return errors[code] || 'অজানা ত্রুটি';
    }

    function showResultsModal(predictions) {
        // Create a simple modal to show results
        const modalHtml = `
            <div class="modal fade" id="resultsModal" tabindex="-1" aria-hidden="true">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header bg-primary text-white">
                            <h5 class="modal-title bangla">পরীক্ষার ফলাফল</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <h5 class="bangla">সম্ভাব্য রোগ:</h5>
                            <ul class="list-group">
                                ${predictions.map(pred => `
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        <span class="bangla">${pred.disease}</span>
                                        <span class="badge bg-primary rounded-pill">${pred.probability}</span>
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary bangla" data-bs-dismiss="modal">বন্ধ করুন</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalHtml);
        const modal = new bootstrap.Modal(document.getElementById('resultsModal'));
        modal.show();

        // Remove modal when closed
        document.getElementById('resultsModal').addEventListener('hidden.bs.modal', function() {
            this.remove();
        });
    }
});

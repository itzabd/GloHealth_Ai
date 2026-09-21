document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('authModal');
    const modalContainer = document.getElementById('authModalContainer');

    if (!modal || !modalContainer) return;

    function getLoginHTML() {
        return `
        <!-- LEFT COLUMN: Brand & Clinical Telemetry Intro (Matches login.html) -->
        <div class="gh-auth-modal__brand">
            <div>
                <div style="margin-bottom: 28px; display: inline-flex; align-items: center; gap: 8px;">
                    <div style="width: 28px; height: 28px; border-radius: 6px; background-color: #0369a1; display: flex; align-items: center; justify-content: center; color: white;">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="12" y1="5" x2="12" y2="19"></line>
                            <line x1="5" y1="12" x2="19" y2="12"></line>
                        </svg>
                    </div>
                    <span class="gh-nav__brand-name" style="font-size: 20px;">GloHealth</span>
                    <span class="gh-nav__brand-badge" style="font-size: 11px;">AI</span>
                </div>

                <h2 class="gh-auth-modal__brand-title">Welcome back.</h2>
                <p class="gh-auth-modal__brand-sub">Sign in to access your symptom history, appointments, and regional insights.</p>

                <ul class="gh-auth-modal__precautions">
                    <li>
                        <span class="material-symbols-outlined">stethoscope</span>
                        <span>AI Diagnostic Screening with clinical-grade accuracy</span>
                    </li>
                    <li>
                        <span class="material-symbols-outlined">event</span>
                        <span>Manage and track appointments with verified specialists</span>
                    </li>
                    <li>
                        <span class="material-symbols-outlined">map</span>
                        <span>Real-time epidemiological telemetry and regional outbreak tracking</span>
                    </li>
                </ul>
            </div>

            <div class="u-mono u-muted" style="font-size: 11.5px; margin-top: 32px;">
                HIPAA compliant · ISO-27799 encrypted
            </div>
        </div>

        <!-- RIGHT COLUMN: Sign In Form (Matches login.html) -->
        <div class="gh-auth-modal__form-col">
            <button type="button" class="gh-auth-modal__close" id="closeAuthModal" aria-label="Close modal">
                <span class="material-symbols-outlined" style="font-size: 18px;">close</span>
            </button>

            <div style="max-width: 380px; width: 100%; margin: 0 auto;">
                <div style="margin-bottom: 22px;">
                    <h3 style="font-size: 22px; font-weight: 700; margin-bottom: 4px; color: var(--gh-text);">Sign In</h3>
                    <p class="u-muted" style="font-size: 13px; margin: 0;">Use your GloHealth account</p>
                </div>

                <form action="/login" method="POST" id="modalLoginForm">
                    <div style="margin-bottom: 15px;">
                        <label for="modalEmail" style="display: block; font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 5px;">Email address</label>
                        <div class="gh-auth__input-wrap">
                            <input type="email" id="modalEmail" name="email" required placeholder="name@example.com" class="gh-auth__input" autofocus autocomplete="email">
                        </div>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <label for="modalPassword" style="display: block; font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 5px;">Password</label>
                        <div class="gh-auth__input-wrap">
                            <input type="password" id="modalPassword" name="password" required placeholder="••••••••" class="gh-auth__input" autocomplete="current-password">
                        </div>
                    </div>

                    <div class="gh-row gh-row--between" style="font-size: 12.5px; margin-bottom: 22px; display: flex; justify-content: space-between; align-items: center;">
                        <label style="display: flex; align-items: center; gap: 7px; cursor: pointer; color: var(--gh-text);">
                            <input type="checkbox" name="remember" style="accent-color: var(--gh-primary); width: 15px; height: 15px; border-radius: 4px;">
                            <span>Remember me</span>
                        </label>
                        <a href="/login" style="color: var(--gh-primary); font-weight: 500; text-decoration: none;">Forgot?</a>
                    </div>

                    <button type="submit" class="gh-btn gh-btn--primary gh-btn--block" style="width: 100%; justify-content: center; padding: 11px; font-size: 14px; font-weight: 600; border-radius: 8px;">
                        Sign In
                    </button>
                </form>

                <div class="gh-auth__divider">
                    <span>or continue with</span>
                </div>

                <div class="gh-grid gh-grid--2" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 22px;">
                    <button type="button" class="gh-btn gh-btn--ghost" style="border: 1px solid var(--gh-border); justify-content: center; font-size: 12.5px; padding: 7px 10px; border-radius: 6px;">
                        Google
                    </button>
                    <button type="button" class="gh-btn gh-btn--ghost" style="border: 1px solid var(--gh-border); justify-content: center; font-size: 12.5px; padding: 7px 10px; border-radius: 6px;">
                        Apple
                    </button>
                </div>

                <div style="text-align: center; font-size: 13px; color: var(--gh-text-muted);">
                    New to GloHealth? <a href="#" data-switch-auth="signup" style="color: var(--gh-primary); font-weight: 600; text-decoration: none;">Create an account</a>
                </div>
            </div>
        </div>
        `;
    }

    function getSignupHTML() {
        return `
        <!-- LEFT COLUMN: Brand & Clinical Telemetry Intro (Matches signup.html) -->
        <div class="gh-auth-modal__brand">
            <div>
                <div style="margin-bottom: 28px; display: inline-flex; align-items: center; gap: 8px;">
                    <div style="width: 28px; height: 28px; border-radius: 6px; background-color: #0369a1; display: flex; align-items: center; justify-content: center; color: white;">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="12" y1="5" x2="12" y2="19"></line>
                            <line x1="5" y1="12" x2="19" y2="12"></line>
                        </svg>
                    </div>
                    <span class="gh-nav__brand-name" style="font-size: 20px;">GloHealth</span>
                    <span class="gh-nav__brand-badge" style="font-size: 11px;">AI</span>
                </div>

                <h2 class="gh-auth-modal__brand-title">Clinical-grade AI healthcare for Bangladesh.</h2>
                <p class="gh-auth-modal__brand-sub">Join patients and healthcare specialists nationwide using predictive diagnostics and real-time surveillance.</p>

                <ul class="gh-auth-modal__precautions">
                    <li>
                        <span class="material-symbols-outlined">vital_signs</span>
                        <span>132+ symptom features modeled on national disease vectors</span>
                    </li>
                    <li>
                        <span class="material-symbols-outlined">verified_user</span>
                        <span>Verified physician network across all 8 administrative divisions</span>
                    </li>
                    <li>
                        <span class="material-symbols-outlined">shield</span>
                        <span>Secure medical profile and historical encounter encryption</span>
                    </li>
                </ul>
            </div>

            <div class="u-mono u-muted" style="font-size: 11.5px; margin-top: 32px;">
                ISO-27799 Encrypted Telemetry · BMDC Aligned
            </div>
        </div>

        <!-- RIGHT COLUMN: Signup Form (Matches signup.html) -->
        <div class="gh-auth-modal__form-col">
            <button type="button" class="gh-auth-modal__close" id="closeAuthModal" aria-label="Close modal">
                <span class="material-symbols-outlined" style="font-size: 18px;">close</span>
            </button>

            <div style="max-width: 420px; width: 100%; margin: 0 auto; padding-top: 10px; padding-bottom: 10px;">
                <div style="margin-bottom: 18px;">
                    <h3 style="font-size: 22px; font-weight: 700; margin-bottom: 4px; color: var(--gh-text);">Create Account</h3>
                    <p class="u-muted" style="font-size: 13px; margin: 0;">Register for predictive clinical access</p>
                </div>

                <form action="/signup" method="POST" id="modalSignupForm">
                    <div style="margin-bottom: 12px;">
                        <label for="modalName" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">Full Name</label>
                        <div class="gh-auth__input-wrap">
                            <input type="text" id="modalName" name="name" required placeholder="e.g. Dr. Sabrina Ahmed" class="gh-auth__input" autofocus autocomplete="name">
                        </div>
                    </div>

                    <div style="margin-bottom: 12px;">
                        <label for="modalSignupEmail" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">Email Address</label>
                        <div class="gh-auth__input-wrap">
                            <input type="email" id="modalSignupEmail" name="email" required placeholder="name@example.com" class="gh-auth__input" autocomplete="email">
                        </div>
                    </div>

                    <div style="margin-bottom: 14px;">
                        <label for="modalSignupPassword" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">Password</label>
                        <div class="gh-auth__input-wrap">
                            <input type="password" id="modalSignupPassword" name="password" required minlength="6" placeholder="••••••••" class="gh-auth__input" autocomplete="new-password">
                        </div>
                        <span class="u-muted" style="font-size: 11px; margin-top: 3px; display: block;">At least 6 characters</span>
                    </div>

                    <!-- Divider Label: Address -->
                    <div class="gh-auth__section-divider">
                        <span>Address Information</span>
                    </div>

                    <div style="margin-bottom: 12px;">
                        <label for="modalAddress1" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">Address Line 1</label>
                        <div class="gh-auth__input-wrap">
                            <input type="text" id="modalAddress1" name="address_line1" required placeholder="House / street" class="gh-auth__input" autocomplete="address-line1">
                        </div>
                    </div>

                    <div style="margin-bottom: 12px;">
                        <label for="modalAddress2" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">Address Line 2 (Optional)</label>
                        <div class="gh-auth__input-wrap">
                            <input type="text" id="modalAddress2" name="address_line2" placeholder="Apartment, floor" class="gh-auth__input" autocomplete="address-line2">
                        </div>
                    </div>

                    <div class="gh-grid gh-grid--2" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
                        <div>
                            <label for="modalCity" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">City</label>
                            <div class="gh-auth__input-wrap">
                                <input type="text" id="modalCity" name="city" required placeholder="e.g. Dhaka" class="gh-auth__input" autocomplete="address-level2">
                            </div>
                        </div>
                        <div>
                            <label for="modalPostal" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">Postal Code</label>
                            <div class="gh-auth__input-wrap">
                                <input type="text" id="modalPostal" name="postal_code" required maxlength="4" placeholder="1207" class="gh-auth__input" autocomplete="postal-code">
                            </div>
                        </div>
                    </div>

                    <div style="margin-bottom: 16px;">
                        <label for="modalDivision" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">Division</label>
                        <div class="gh-auth__input-wrap">
                            <select id="modalDivision" name="division" required class="gh-auth__input" style="cursor: pointer;">
                                <option value="" disabled selected>Select your division</option>
                                <option value="Dhaka">Dhaka</option>
                                <option value="Chittagong">Chittagong</option>
                                <option value="Rajshahi">Rajshahi</option>
                                <option value="Khulna">Khulna</option>
                                <option value="Barisal">Barisal</option>
                                <option value="Sylhet">Sylhet</option>
                                <option value="Rangpur">Rangpur</option>
                                <option value="Mymensingh">Mymensingh</option>
                            </select>
                        </div>
                    </div>

                    <div style="margin-bottom: 18px;">
                        <label style="display: flex; align-items: flex-start; gap: 7px; cursor: pointer; font-size: 12px; color: var(--gh-text);">
                            <input type="checkbox" required style="accent-color: var(--gh-primary); width: 15px; height: 15px; border-radius: 4px; margin-top: 2px;">
                            <span>I agree to the <a href="/plans" style="color: var(--gh-primary); text-decoration: none;">Terms of Service</a> and <a href="/plans" style="color: var(--gh-primary); text-decoration: none;">Privacy Policy</a></span>
                        </label>
                    </div>

                    <button type="submit" class="gh-btn gh-btn--primary gh-btn--block" style="width: 100%; justify-content: center; padding: 11px; font-size: 14px; font-weight: 600; border-radius: 8px;">
                        Create Account
                    </button>
                </form>

                <div style="text-align: center; font-size: 13px; color: var(--gh-text-muted); margin-top: 18px;">
                    Already have an account? <a href="#" data-switch-auth="login" style="color: var(--gh-primary); font-weight: 600; text-decoration: none;">Sign in</a>
                </div>
            </div>
        </div>
        `;
    }

    function renderView(view) {
        if (view === 'signup') {
            modalContainer.innerHTML = getSignupHTML();
        } else {
            modalContainer.innerHTML = getLoginHTML();
        }

        // Attach close button
        const closeBtn = document.getElementById('closeAuthModal');
        if (closeBtn) {
            closeBtn.addEventListener('click', closeModal);
        }

        // Attach switch links
        modalContainer.querySelectorAll('[data-switch-auth]').forEach(function(link) {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                renderView(link.dataset.switchAuth);
            });
        });

        // Auto-focus first input
        const firstInput = modalContainer.querySelector('input');
        if (firstInput) {
            setTimeout(function() { firstInput.focus(); }, 60);
        }
    }

    function openModal(view) {
        renderView(view || 'signup');
        modal.classList.add('is-open');
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        modal.classList.remove('is-open');
        document.body.style.overflow = '';
    }

    // Bind triggers from navbar and hero buttons
    document.querySelectorAll('[data-auth-open]').forEach(function(btn) {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            openModal(btn.dataset.authOpen);
        });
    });

    // Close on backdrop click outside container
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeModal();
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal.classList.contains('is-open')) {
            closeModal();
        }
    });
});

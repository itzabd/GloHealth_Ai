(function() {
    let modal = null;
    let modalContainer = null;
    let currentView = 'signup';

    function t(key, fallback) {
        var lang = window.GH_LANG || localStorage.getItem('glohealth_lang') || 'en';
        var dict = (window.GH_I18N && window.GH_I18N[lang]) || {};
        if (dict[key] !== undefined) return dict[key];
        var enDict = (window.GH_I18N && window.GH_I18N['en']) || {};
        if (enDict[key] !== undefined) return enDict[key];
        return fallback !== undefined ? fallback : key;
    }

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

                <h2 class="gh-auth-modal__brand-title">${t('a.welcome_title', 'Welcome back.')}</h2>
                <p class="gh-auth-modal__brand-sub">${t('a.welcome_sub', 'Sign in to access your symptom history, appointments, and regional insights.')}</p>

                <ul class="gh-auth-modal__precautions">
                    <li>
                        <span class="material-symbols-outlined">stethoscope</span>
                        <span>${t('a.feat_diag', 'AI Diagnostic Screening with clinical-grade accuracy')}</span>
                    </li>
                    <li>
                        <span class="material-symbols-outlined">event</span>
                        <span>${t('a.feat_appts', 'Manage and track appointments with verified specialists')}</span>
                    </li>
                    <li>
                        <span class="material-symbols-outlined">map</span>
                        <span>${t('a.feat_telemetry', 'Real-time epidemiological telemetry and regional outbreak tracking')}</span>
                    </li>
                </ul>
            </div>

            <div class="u-mono u-muted" style="font-size: 11.5px; margin-top: 32px;">
                ${t('a.hipaa', 'HIPAA compliant · ISO-27799 encrypted')}
            </div>
        </div>

        <!-- RIGHT COLUMN: Sign In Form (Matches login.html) -->
        <div class="gh-auth-modal__form-col">
            <button type="button" class="gh-auth-modal__close" id="closeAuthModal" aria-label="Close modal">
                <span class="material-symbols-outlined" style="font-size: 18px;">close</span>
            </button>

            <div style="max-width: 380px; width: 100%; margin: 0 auto;">
                <div style="margin-bottom: 22px;">
                    <h3 style="font-size: 22px; font-weight: 700; margin-bottom: 4px; color: var(--gh-text);">${t('a.login_title', 'Sign In')}</h3>
                    <p class="u-muted" style="font-size: 13px; margin: 0;">${t('a.login_sub', 'Use your GloHealth account')}</p>
                </div>

                <form action="/login" method="POST" id="modalLoginForm">
                    <div style="margin-bottom: 15px;">
                        <label for="modalEmail" style="display: block; font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 5px;">${t('a.email', 'Email address')}</label>
                        <div class="gh-auth__input-wrap">
                            <input type="email" id="modalEmail" name="email" required placeholder="${t('a.ph_email', 'name@example.com')}" class="gh-auth__input" autofocus autocomplete="email">
                        </div>
                    </div>

                    <div style="margin-bottom: 15px;">
                        <label for="modalPassword" style="display: block; font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 5px;">${t('a.password', 'Password')}</label>
                        <div class="gh-auth__input-wrap">
                            <input type="password" id="modalPassword" name="password" required placeholder="••••••••" class="gh-auth__input" autocomplete="current-password">
                        </div>
                    </div>

                    <div class="gh-row gh-row--between" style="font-size: 12.5px; margin-bottom: 22px; display: flex; justify-content: space-between; align-items: center;">
                        <label style="display: flex; align-items: center; gap: 7px; cursor: pointer; color: var(--gh-text);">
                            <input type="checkbox" name="remember" style="accent-color: var(--gh-primary); width: 15px; height: 15px; border-radius: 4px;">
                            <span>${t('a.remember', 'Remember me')}</span>
                        </label>
                        <a href="/login" style="color: var(--gh-primary); font-weight: 500; text-decoration: none;">${t('a.forgot', 'Forgot?')}</a>
                    </div>

                    <button type="submit" class="gh-btn gh-btn--primary gh-btn--block" style="width: 100%; justify-content: center; padding: 11px; font-size: 14px; font-weight: 600; border-radius: 8px;">
                        ${t('a.login_title', 'Sign In')}
                    </button>
                </form>

                <div class="gh-auth__divider">
                    <span>${t('a.or', 'or continue with')}</span>
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
                    ${t('a.new', 'New to GloHealth?')} <a href="#" data-switch-auth="signup" style="color: var(--gh-primary); font-weight: 600; text-decoration: none;">${t('a.create', 'Create an account')}</a>
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

                <h2 class="gh-auth-modal__brand-title">${t('a.brand_title_signup', 'Clinical-grade AI healthcare for Bangladesh.')}</h2>
                <p class="gh-auth-modal__brand-sub">${t('a.brand_sub_signup', 'Join patients and healthcare specialists nationwide using predictive diagnostics and real-time surveillance.')}</p>

                <ul class="gh-auth-modal__precautions">
                    <li>
                        <span class="material-symbols-outlined">vital_signs</span>
                        <span>${t('a.feat_sym_count', '132+ symptom features modeled on national disease vectors')}</span>
                    </li>
                    <li>
                        <span class="material-symbols-outlined">verified_user</span>
                        <span>${t('a.feat_network', 'Verified physician network across all 8 administrative divisions')}</span>
                    </li>
                    <li>
                        <span class="material-symbols-outlined">shield</span>
                        <span>${t('a.feat_secure', 'Secure medical profile and historical encounter encryption')}</span>
                    </li>
                </ul>
            </div>

            <div class="u-mono u-muted" style="font-size: 11.5px; margin-top: 32px;">
                ${t('a.badge_signup', 'ISO-27799 Encrypted Telemetry · BMDC Aligned')}
            </div>
        </div>

        <!-- RIGHT COLUMN: Signup Form (Matches signup.html) -->
        <div class="gh-auth-modal__form-col">
            <button type="button" class="gh-auth-modal__close" id="closeAuthModal" aria-label="Close modal">
                <span class="material-symbols-outlined" style="font-size: 18px;">close</span>
            </button>

            <div style="max-width: 420px; width: 100%; margin: 0 auto; padding-top: 10px; padding-bottom: 10px;">
                <div style="margin-bottom: 18px;">
                    <h3 style="font-size: 22px; font-weight: 700; margin-bottom: 4px; color: var(--gh-text);">${t('a.signup_title', 'Create Account')}</h3>
                    <p class="u-muted" style="font-size: 13px; margin: 0;">${t('a.signup_sub', 'Register for predictive clinical access')}</p>
                </div>

                <form action="/signup" method="POST" id="modalSignupForm">
                    <div style="margin-bottom: 12px;">
                        <label for="modalName" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">${t('a.name', 'Full Name')}</label>
                        <div class="gh-auth__input-wrap">
                            <input type="text" id="modalName" name="name" required placeholder="${t('a.ph_name', 'e.g. Dr. Sabrina Ahmed')}" class="gh-auth__input" autofocus autocomplete="name">
                        </div>
                    </div>

                    <div style="margin-bottom: 12px;">
                        <label for="modalSignupEmail" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">${t('a.email', 'Email Address')}</label>
                        <div class="gh-auth__input-wrap">
                            <input type="email" id="modalSignupEmail" name="email" required placeholder="${t('a.ph_email', 'name@example.com')}" class="gh-auth__input" autocomplete="email">
                        </div>
                    </div>

                    <div style="margin-bottom: 14px;">
                        <label for="modalSignupPassword" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">${t('a.password', 'Password')}</label>
                        <div class="gh-auth__input-wrap">
                            <input type="password" id="modalSignupPassword" name="password" required minlength="6" placeholder="••••••••" class="gh-auth__input" autocomplete="new-password">
                        </div>
                        <span class="u-muted" style="font-size: 11px; margin-top: 3px; display: block;">${t('a.pwd_min', 'At least 6 characters')}</span>
                    </div>

                    <!-- Divider Label: Address -->
                    <div class="gh-auth__section-divider">
                        <span>${t('a.addr', 'Address Information')}</span>
                    </div>

                    <div style="margin-bottom: 12px;">
                        <label for="modalAddress1" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">${t('a.addr1', 'Address Line 1')}</label>
                        <div class="gh-auth__input-wrap">
                            <input type="text" id="modalAddress1" name="address_line1" required placeholder="${t('a.ph_addr1', 'House / street')}" class="gh-auth__input" autocomplete="address-line1">
                        </div>
                    </div>

                    <div style="margin-bottom: 12px;">
                        <label for="modalAddress2" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">${t('a.addr2', 'Address Line 2 (Optional)')}</label>
                        <div class="gh-auth__input-wrap">
                            <input type="text" id="modalAddress2" name="address_line2" placeholder="${t('a.ph_addr2', 'Apartment, floor')}" class="gh-auth__input" autocomplete="address-line2">
                        </div>
                    </div>

                    <div class="gh-grid gh-grid--2" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
                        <div>
                            <label for="modalCity" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">${t('a.city', 'City')}</label>
                            <div class="gh-auth__input-wrap">
                                <input type="text" id="modalCity" name="city" required placeholder="${t('a.ph_city', 'e.g. Dhaka')}" class="gh-auth__input" autocomplete="address-level2">
                            </div>
                        </div>
                        <div>
                            <label for="modalPostal" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">${t('a.postal', 'Postal Code')}</label>
                            <div class="gh-auth__input-wrap">
                                <input type="text" id="modalPostal" name="postal_code" required maxlength="4" placeholder="${t('a.ph_postal', '1207')}" class="gh-auth__input" autocomplete="postal-code">
                            </div>
                        </div>
                    </div>

                    <div style="margin-bottom: 16px;">
                        <label for="modalDivision" style="display: block; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--gh-text-muted); margin-bottom: 4px;">${t('a.division', 'Division')}</label>
                        <div class="gh-auth__input-wrap">
                            <select id="modalDivision" name="division" required class="gh-auth__input" style="cursor: pointer;">
                                <option value="" disabled selected>${t('a.select_div', 'Select your division')}</option>
                                <option value="Dhaka">${t('a.div_dhaka', 'Dhaka')}</option>
                                <option value="Chittagong">${t('a.div_chittagong', 'Chittagong')}</option>
                                <option value="Rajshahi">${t('a.div_rajshahi', 'Rajshahi')}</option>
                                <option value="Khulna">${t('a.div_khulna', 'Khulna')}</option>
                                <option value="Barisal">${t('a.div_barisal', 'Barisal')}</option>
                                <option value="Sylhet">${t('a.div_sylhet', 'Sylhet')}</option>
                                <option value="Rangpur">${t('a.div_rangpur', 'Rangpur')}</option>
                                <option value="Mymensingh">${t('a.div_mymensingh', 'Mymensingh')}</option>
                            </select>
                        </div>
                    </div>

                    <div style="margin-bottom: 18px;">
                        <label style="display: flex; align-items: flex-start; gap: 7px; cursor: pointer; font-size: 12px; color: var(--gh-text);">
                            <input type="checkbox" required style="accent-color: var(--gh-primary); width: 15px; height: 15px; border-radius: 4px; margin-top: 2px;">
                            <span>${t('a.agree_prefix', 'I agree to the')} <a href="/plans" style="color: var(--gh-primary); text-decoration: none;">${t('a.terms_link', 'Terms of Service')}</a> ${t('a.and', 'and')} <a href="/plans" style="color: var(--gh-primary); text-decoration: none;">${t('a.privacy_link', 'Privacy Policy')}</a></span>
                        </label>
                    </div>

                    <button type="submit" class="gh-btn gh-btn--primary gh-btn--block" style="width: 100%; justify-content: center; padding: 11px; font-size: 14px; font-weight: 600; border-radius: 8px;">
                        ${t('a.signup_title', 'Create Account')}
                    </button>
                </form>

                <div style="text-align: center; font-size: 13px; color: var(--gh-text-muted); margin-top: 18px;">
                    ${t('a.have', 'Already have an account?')} <a href="#" data-switch-auth="login" style="color: var(--gh-primary); font-weight: 600; text-decoration: none;">${t('a.signin_link', 'Sign in')}</a>
                </div>
            </div>
        </div>
        `;
    }

    function renderView(view) {
        modal = modal || document.getElementById('authModal');
        modalContainer = modalContainer || document.getElementById('authModalContainer');
        if (!modalContainer) return;

        currentView = view || 'signup';
        if (currentView === 'signup') {
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

        // Form submit feedback & anti-duplicate submit (UI/UX Pro Max)
        const form = modalContainer.querySelector('form');
        if (form) {
            form.addEventListener('submit', function() {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.classList.add('gh-btn--loading');
                    submitBtn.setAttribute('aria-busy', 'true');
                    setTimeout(function() {
                        submitBtn.disabled = true;
                    }, 50);
                }
            });
        }

        // Auto-focus first input
        const firstInput = modalContainer.querySelector('input');
        if (firstInput) {
            setTimeout(function() { firstInput.focus(); }, 60);
        }
    }

    function openModal(view) {
        modal = modal || document.getElementById('authModal');
        modalContainer = modalContainer || document.getElementById('authModalContainer');
        if (!modal || !modalContainer) return;

        // Dismiss mobile nav drawer and its backdrop so modal is never obscured
        var navDrawer = document.getElementById('navDrawer');
        var navBackdrop = document.getElementById('navBackdrop');
        var navToggle = document.getElementById('navToggle');
        var navToggleIcon = document.getElementById('navToggleIcon');
        if (navDrawer) {
            navDrawer.classList.remove('is-open');
            navDrawer.setAttribute('aria-hidden', 'true');
        }
        if (navBackdrop) {
            navBackdrop.classList.remove('is-open');
        }
        if (navToggle) {
            navToggle.setAttribute('aria-expanded', 'false');
        }
        if (navToggleIcon) {
            navToggleIcon.textContent = 'menu';
        }

        renderView(view || 'signup');
        modal.classList.add('is-open');
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        modal = modal || document.getElementById('authModal');
        if (!modal) return;
        modal.classList.remove('is-open');
        document.body.style.overflow = '';
    }

    // Expose globally so inline onclick or external scripts can call them directly
    window.GH_OPEN_AUTH = openModal;
    window.GH_CLOSE_AUTH = closeModal;

    function initAuthModal() {
        modal = document.getElementById('authModal');
        modalContainer = document.getElementById('authModalContainer');
        if (!modal) return;

        // Close on backdrop click outside container
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeModal();
            }
        });

        // Close on Escape key & Focus Trap
        document.addEventListener('keydown', function(e) {
            if (!modal.classList.contains('is-open')) return;

            if (e.key === 'Escape') {
                closeModal();
                return;
            }

            // Keyboard Focus Trap (WCAG 2.1 AAA)
            if (e.key === 'Tab') {
                const focusable = modalContainer.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
                if (focusable.length === 0) return;
                const firstEl = focusable[0];
                const lastEl = focusable[focusable.length - 1];

                if (e.shiftKey) {
                    if (document.activeElement === firstEl) {
                        e.preventDefault();
                        lastEl.focus();
                    }
                } else {
                    if (document.activeElement === lastEl) {
                        e.preventDefault();
                        firstEl.focus();
                    }
                }
            }
        });

        // Re-render modal when language changes dynamically
        document.addEventListener('gh:langchange', function() {
            if (modal.classList.contains('is-open')) {
                renderView(currentView);
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAuthModal);
    } else {
        initAuthModal();
    }

    // Global delegated listener for any auth modal triggers (active immediately)
    document.addEventListener('click', function(e) {
        var trigger = e.target.closest('[data-auth-open]');
        if (trigger) {
            e.preventDefault();
            e.stopPropagation();
            openModal(trigger.getAttribute('data-auth-open') || 'signup');
        }
    });

})();

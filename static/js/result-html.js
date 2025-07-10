/**
 * Results Page Carousel and Tab Management
 * Handles all interactive elements in the redesigned results page
 */

class ResultsPageManager {
    constructor() {
        this.carousels = new Map();
        this.currentSlides = new Map();
        this.init();
    }

    init() {
        this.initCarousels();
        this.initTabs();
        this.initAnimations();
        
        // Initialize on DOM ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.reinitialize());
        } else {
            this.reinitialize();
        }
    }

    reinitialize() {
        // Reinitialize everything after DOM is fully loaded
        setTimeout(() => {
            this.initCarousels();
            this.initTabs();
            this.addKeyboardNavigation();
            this.addTouchSupport();
        }, 100);
    }

    /**
     * Initialize all carousels on the page
     */
    initCarousels() {
        const carouselContainers = document.querySelectorAll('.carousel-container');
        
        carouselContainers.forEach(container => {
            const carouselId = container.id;
            if (!carouselId) return;

            const slides = container.querySelectorAll('.carousel-slide');
            if (slides.length === 0) return;

            // Initialize carousel state
            this.carousels.set(carouselId, {
                container,
                slides,
                currentIndex: 0,
                totalSlides: slides.length
            });

            this.currentSlides.set(carouselId, 0);

            // Show first slide
            this.showSlide(carouselId, 0);

            // Add navigation event listeners
            this.addCarouselNavigation(carouselId);

            // Add auto-advance for single-slide carousels
            if (slides.length > 1) {
                this.addAutoAdvance(carouselId);
            }
        });
    }

    /**
     * Add navigation controls to a carousel
     */
    addCarouselNavigation(carouselId) {
        const carousel = this.carousels.get(carouselId);
        if (!carousel) return;

        // Find navigation buttons
        const prevBtn = document.querySelector(`[data-carousel="${carouselId}"].prev`);
        const nextBtn = document.querySelector(`[data-carousel="${carouselId}"].next`);

        // Alternative selector for carousel buttons within the same container
        const carouselSection = carousel.container.closest('.analysis-carousel') || 
                               carousel.container.closest('.result-card');
        
        const altPrevBtn = carouselSection?.querySelector('.carousel-btn.prev');
        const altNextBtn = carouselSection?.querySelector('.carousel-btn.next');

        const prevButton = prevBtn || altPrevBtn;
        const nextButton = nextBtn || altNextBtn;

        if (prevButton) {
            prevButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.previousSlide(carouselId);
            });
        }

        if (nextButton) {
            nextButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.nextSlide(carouselId);
            });
        }

        // Update button states
        this.updateNavigationButtons(carouselId);
    }

    /**
     * Show a specific slide in a carousel
     */
    showSlide(carouselId, index) {
        const carousel = this.carousels.get(carouselId);
        if (!carousel) return;

        const { slides, totalSlides } = carousel;
        
        // Ensure index is within bounds
        index = Math.max(0, Math.min(index, totalSlides - 1));
        
        // Hide all slides
        slides.forEach(slide => {
            slide.classList.remove('active');
            slide.style.display = 'none';
        });

        // Show current slide
        if (slides[index]) {
            slides[index].classList.add('active');
            slides[index].style.display = 'block';
            
            // Add entrance animation
            slides[index].style.animation = 'fadeInUp 0.5s ease-out';
        }

        // Update current index
        this.currentSlides.set(carouselId, index);
        carousel.currentIndex = index;

        // Update navigation buttons
        this.updateNavigationButtons(carouselId);
    }

    /**
     * Go to next slide
     */
    nextSlide(carouselId) {
        const currentIndex = this.currentSlides.get(carouselId) || 0;
        const carousel = this.carousels.get(carouselId);
        
        if (carousel) {
            const nextIndex = (currentIndex + 1) % carousel.totalSlides;
            this.showSlide(carouselId, nextIndex);
        }
    }

    /**
     * Go to previous slide
     */
    previousSlide(carouselId) {
        const currentIndex = this.currentSlides.get(carouselId) || 0;
        const carousel = this.carousels.get(carouselId);
        
        if (carousel) {
            const prevIndex = currentIndex === 0 ? carousel.totalSlides - 1 : currentIndex - 1;
            this.showSlide(carouselId, prevIndex);
        }
    }

    /**
     * Update navigation button states
     */
    updateNavigationButtons(carouselId) {
        const carousel = this.carousels.get(carouselId);
        if (!carousel) return;

        const currentIndex = this.currentSlides.get(carouselId) || 0;
        const { totalSlides } = carousel;

        // Find buttons
        const carouselSection = carousel.container.closest('.analysis-carousel') || 
                               carousel.container.closest('.result-card');
        
        const prevBtn = carouselSection?.querySelector('.carousel-btn.prev');
        const nextBtn = carouselSection?.querySelector('.carousel-btn.next');

        // Update button states (for linear navigation)
        if (prevBtn) {
            prevBtn.style.opacity = currentIndex === 0 ? '0.5' : '1';
            prevBtn.style.pointerEvents = currentIndex === 0 ? 'none' : 'auto';
        }

        if (nextBtn) {
            nextBtn.style.opacity = currentIndex === totalSlides - 1 ? '0.5' : '1';
            nextBtn.style.pointerEvents = currentIndex === totalSlides - 1 ? 'none' : 'auto';
        }

        // For circular navigation, always enable buttons
        if (totalSlides > 1) {
            if (prevBtn) {
                prevBtn.style.opacity = '1';
                prevBtn.style.pointerEvents = 'auto';
            }
            if (nextBtn) {
                nextBtn.style.opacity = '1';
                nextBtn.style.pointerEvents = 'auto';
            }
        }
    }

    /**
     * Add auto-advance functionality (optional)
     */
    addAutoAdvance(carouselId, interval = 100000) {
        const carousel = this.carousels.get(carouselId);
        if (!carousel || carousel.totalSlides <= 1) return;

        // Clear any existing interval
        if (carousel.autoInterval) {
            clearInterval(carousel.autoInterval);
        }

        // Set up auto-advance (paused on hover)
        let isPaused = false;

        carousel.autoInterval = setInterval(() => {
            if (!isPaused) {
                this.nextSlide(carouselId);
            }
        }, interval);

        // Pause on hover
        carousel.container.addEventListener('mouseenter', () => {
            isPaused = true;
        });

        carousel.container.addEventListener('mouseleave', () => {
            isPaused = false;
        });

        // Store interval reference
        carousel.autoInterval = carousel.autoInterval;
    }

    /**
     * Initialize tab functionality
     */
    initTabs() {
        // Handle analysis tabs (audio spectrograms, etc.)
        this.initAnalysisTabs();
        
        // Handle face detection tabs
        this.initFaceTabs();
        
        // Handle multi-detection tabs
        this.initMultiDetectionTabs();
    }

    /**
     * Initialize analysis tabs (audio spectrograms, etc.)
     */
    initAnalysisTabs() {
        const tabNavigation = document.querySelector('.analysis-tabs .tab-navigation');
        if (!tabNavigation) return;

        const tabButtons = tabNavigation.querySelectorAll('.tab-btn');
        const tabPanels = document.querySelectorAll('.analysis-tabs .tab-panel');

        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const targetTab = button.getAttribute('data-tab');
                this.switchTab(tabButtons, tabPanels, button, targetTab);
            });
        });
    }

    /**
     * Initialize face detection tabs
     */
    initFaceTabs() {
        const faceTabsContainer = document.querySelector('.face-detection-results');
        if (!faceTabsContainer) return;

        const tabNavigation = faceTabsContainer.querySelector('.tab-navigation');
        if (!tabNavigation) return;

        const tabButtons = tabNavigation.querySelectorAll('.tab-btn');
        const tabPanels = faceTabsContainer.querySelectorAll('.tab-panel');

        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const targetTab = button.getAttribute('data-tab');
                this.switchTab(tabButtons, tabPanels, button, targetTab);
            });
        });
    }

    /**
     * Initialize multi-detection tabs
     */
    initMultiDetectionTabs() {
        const multiDetectionContainer = document.querySelector('.multi-detection-results');
        if (!multiDetectionContainer) return;

        const tabNavigation = multiDetectionContainer.querySelector('.tab-navigation');
        if (!tabNavigation) return;

        const tabButtons = tabNavigation.querySelectorAll('.tab-btn');
        const tabPanels = multiDetectionContainer.querySelectorAll('.tab-panel');

        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const targetTab = button.getAttribute('data-tab');
                this.switchTab(tabButtons, tabPanels, button, targetTab);
            });
        });
    }

    /**
     * Switch between tabs
     */
    switchTab(tabButtons, tabPanels, activeButton, targetTabId) {
        // Remove active class from all buttons
        tabButtons.forEach(btn => btn.classList.remove('active'));
        
        // Add active class to clicked button
        activeButton.classList.add('active');

        // Hide all tab panels
        tabPanels.forEach(panel => {
            panel.classList.remove('active');
            panel.style.display = 'none';
        });

        // Show target tab panel
        const targetPanel = document.getElementById(targetTabId);
        if (targetPanel) {
            targetPanel.classList.add('active');
            targetPanel.style.display = 'block';
            
            // Add entrance animation
            targetPanel.style.animation = 'fadeInUp 0.5s ease-out';
        }

        // Reinitialize carousels in the new tab
        setTimeout(() => {
            this.initCarousels();
        }, 100);
    }

    /**
     * Add keyboard navigation support
     */
    addKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // Only handle keyboard navigation if focus is on carousel or no input is focused
            const focusedElement = document.activeElement;
            const isInputFocused = focusedElement.tagName === 'INPUT' || 
                                 focusedElement.tagName === 'TEXTAREA' || 
                                 focusedElement.isContentEditable;

            if (isInputFocused) return;

            switch (e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.handleKeyboardNavigation('prev');
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.handleKeyboardNavigation('next');
                    break;
                case 'Tab':
                    // Allow tab navigation between carousel controls
                    break;
            }
        });
    }

    /**
     * Handle keyboard navigation for active carousel
     */
    handleKeyboardNavigation(direction) {
        // Find the currently visible carousel
        const visibleCarousel = Array.from(this.carousels.keys()).find(carouselId => {
            const carousel = this.carousels.get(carouselId);
            const container = carousel.container;
            const rect = container.getBoundingClientRect();
            return rect.top >= 0 && rect.bottom <= window.innerHeight;
        });

        if (visibleCarousel) {
            if (direction === 'next') {
                this.nextSlide(visibleCarousel);
            } else if (direction === 'prev') {
                this.previousSlide(visibleCarousel);
            }
        }
    }

    /**
     * Add touch/swipe support for mobile
     */
    addTouchSupport() {
        this.carousels.forEach((carousel, carouselId) => {
            let startX = 0;
            let endX = 0;
            const minSwipeDistance = 50;

            carousel.container.addEventListener('touchstart', (e) => {
                startX = e.touches[0].clientX;
            }, { passive: true });

            carousel.container.addEventListener('touchend', (e) => {
                endX = e.changedTouches[0].clientX;
                const swipeDistance = Math.abs(endX - startX);

                if (swipeDistance > minSwipeDistance) {
                    if (endX < startX) {
                        // Swipe left - next slide
                        this.nextSlide(carouselId);
                    } else if (endX > startX) {
                        // Swipe right - previous slide
                        this.previousSlide(carouselId);
                    }
                }
            }, { passive: true });
        });
    }

    /**
     * Initialize entrance animations
     */
    initAnimations() {
        // Intersection Observer for scroll-triggered animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, observerOptions);

        // Observe all result cards
        document.querySelectorAll('.result-card').forEach(card => {
            observer.observe(card);
        });
    }

    /**
     * Cleanup function
     */
    destroy() {
        // Clear all auto-advance intervals
        this.carousels.forEach(carousel => {
            if (carousel.autoInterval) {
                clearInterval(carousel.autoInterval);
            }
        });

        // Clear maps
        this.carousels.clear();
        this.currentSlides.clear();
    }
}

// Global function for back button (used in templates)
function deleteFilesAndGoBack() {
    // Add a smooth transition before going back
    const resultsMain = document.querySelector('.results-main');
    if (resultsMain) {
        resultsMain.style.transition = 'opacity 0.3s ease-out, transform 0.3s ease-out';
        resultsMain.style.opacity = '0';
        resultsMain.style.transform = 'translateY(-20px)';
    }

    // Go back after animation
    setTimeout(() => {
        window.history.back();
    }, 300);
}

// Initialize the results page manager
let resultsPageManager;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        resultsPageManager = new ResultsPageManager();
    });
} else {
    resultsPageManager = new ResultsPageManager();
}

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Pause auto-advance when page is not visible
        if (resultsPageManager) {
            resultsPageManager.carousels.forEach(carousel => {
                if (carousel.autoInterval) {
                    carousel.wasPaused = true;
                    clearInterval(carousel.autoInterval);
                }
            });
        }
    } else {
        // Resume auto-advance when page becomes visible
        if (resultsPageManager) {
            resultsPageManager.carousels.forEach((carousel, carouselId) => {
                if (carousel.wasPaused && carousel.totalSlides > 1) {
                    resultsPageManager.addAutoAdvance(carouselId);
                    carousel.wasPaused = false;
                }
            });
        }
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (resultsPageManager) {
        resultsPageManager.destroy();
    }
});

// Export for potential external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ResultsPageManager, deleteFilesAndGoBack };
}